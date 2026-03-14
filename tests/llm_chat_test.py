# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for ChatSession orchestration and the public LLM chat API."""

import asyncio
from types import SimpleNamespace
import unittest

from trillim.events import ChatDoneEvent, ChatFinalTextEvent, ChatSearchResultEvent
from trillim.harnesses._base import Harness
from trillim.harnesses._default import DefaultHarness
from trillim.harnesses._search import SearchHarness
from trillim.server import LLM
from trillim.server._models import ServerState


class _FakeTokenizer:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(chr(token_id) for token_id in token_ids)


class _FinalizingRewriteTokenizer(_FakeTokenizer):
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        if not add_generation_prompt and messages and messages[-1]["role"] == "assistant":
            return "rewritten-final-output"
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class _ScriptedEngine:
    def __init__(self, responses: list[str], *, tokenizer=None, max_context_tokens: int = 4096):
        self._responses = list(responses)
        self.tokenizer = tokenizer or _FakeTokenizer()
        self.arch_config = SimpleNamespace(max_position_embeddings=max_context_tokens)
        self._cached_prompt_str = ""
        self._last_cache_hit = 3
        self.finalized_prompt_snapshots = []

    @property
    def cached_prompt_str(self) -> str:
        return self._cached_prompt_str

    @property
    def last_cache_hit(self) -> int:
        return self._last_cache_hit

    def finalize_prompt_cache(self, snapshot) -> None:
        self.finalized_prompt_snapshots.append(snapshot)
        self._cached_prompt_str = snapshot.prompt_str or ""

    def reset_prompt_cache(self) -> None:
        self._cached_prompt_str = ""
        self._last_cache_hit = 0

    async def generate(self, **_):
        response = self._responses.pop(0)
        for ch in response:
            yield ord(ch)


class _SlowScriptedEngine(_ScriptedEngine):
    async def generate(self, **kwargs):
        await asyncio.sleep(0.05)
        async for token in super().generate(**kwargs):
            yield token


class _TokenDelayScriptedEngine(_ScriptedEngine):
    def __init__(
        self,
        responses: list[str],
        *,
        token_delays: list[float],
        tokenizer=None,
        max_context_tokens: int = 4096,
    ):
        super().__init__(
            responses,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
        )
        self._token_delays = list(token_delays)

    async def generate(self, **_):
        response = self._responses.pop(0)
        for index, ch in enumerate(response):
            if index < len(self._token_delays):
                await asyncio.sleep(self._token_delays[index])
            yield ord(ch)


class _SuccessfulSearch:
    def __init__(self, results: str):
        self._results = results

    async def search(self, query: str) -> str:
        self.last_query = query
        return self._results


class _UnavailableSearch:
    async def search(self, query: str) -> str:
        from trillim.harnesses._search_utils import SearchError

        raise SearchError(f"Search unavailable: {query}")


class _HarnessProbe(Harness):
    async def stream_events(self, session, **sampling):
        yield ChatFinalTextEvent(text="done")


class HarnessEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_base_abstract_stream_events_is_empty_async_generator(self):
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["hi"])
        llm.harness = _HarnessProbe(llm.engine)
        session = llm.session([{"role": "user", "content": "x"}])

        self.assertIs(llm.harness.arch_config, llm.engine.arch_config)
        events = [event async for event in Harness.stream_events(llm.harness, session)]

        self.assertEqual(events, [None])

    async def test_default_harness_emits_token_and_final_text_events(self):
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["hi"])
        llm.harness = DefaultHarness(llm.engine)
        session = llm.session([{"role": "user", "content": "hello"}])

        events = [event async for event in llm.harness.stream_events(session)]

        self.assertEqual([event.type for event in events], ["token", "token", "final_text"])
        self.assertEqual("".join(event.text for event in events[:-1]), "hi")
        self.assertIsInstance(events[-1], ChatFinalTextEvent)
        self.assertEqual(events[-1].text, "hi")
        self.assertEqual(
            session.messages,
            (
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ),
        )
        self.assertEqual(llm.engine.cached_prompt_str, "user: hello\nassistant: hi")

    async def test_default_harness_renders_empty_generation_prompt(self):
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["unused"])
        llm.harness = DefaultHarness(llm.engine)
        session = llm.session()

        self.assertEqual(session._render_prompt(add_generation_prompt=True), "assistant: ")

    async def test_search_harness_emits_structured_search_events(self):
        llm = LLM("models/fake", harness_name="search")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["<search>cats</search>", "answer"])
        llm.harness = SearchHarness(llm.engine)
        llm.harness._search = _SuccessfulSearch("curated cat result")
        session = llm.session([{"role": "user", "content": "Find cats"}])

        events = [event async for event in llm.harness.stream_events(session)]

        self.assertEqual(events[0].type, "search_started")
        self.assertEqual(events[0].query, "cats")
        self.assertIsInstance(events[1], ChatSearchResultEvent)
        self.assertTrue(events[1].available)
        self.assertEqual(events[1].content, "curated cat result")
        self.assertEqual("".join(event.text for event in events if event.type == "token"), "answer")
        self.assertEqual(events[-1].type, "final_text")
        self.assertEqual(events[-1].text, "answer")
        self.assertEqual(
            session.messages,
            (
                {"role": "user", "content": "Find cats"},
                {"role": "assistant", "content": "<search>cats</search>"},
                {"role": "search", "content": "curated cat result"},
                {"role": "assistant", "content": "answer"},
            ),
        )

    async def test_search_harness_marks_unavailable_searches(self):
        llm = LLM("models/fake", harness_name="search")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["<search>cats</search>", "fallback"])
        llm.harness = SearchHarness(llm.engine)
        llm.harness._search = _UnavailableSearch()
        session = llm.session([{"role": "user", "content": "Find cats"}])

        events = [event async for event in llm.harness.stream_events(session)]

        self.assertEqual(events[0].type, "search_started")
        self.assertIsInstance(events[1], ChatSearchResultEvent)
        self.assertFalse(events[1].available)
        self.assertIn("Search unavailable", events[1].content)
        self.assertEqual(events[-1].text, "fallback")
        self.assertEqual(
            session.messages,
            (
                {"role": "user", "content": "Find cats"},
                {"role": "assistant", "content": "<search>cats</search>"},
                {
                    "role": "search",
                    "content": "Search unavailable, please answer from your knowledge.",
                },
                {"role": "assistant", "content": "fallback"},
            ),
        )

    async def test_harness_run_remains_plain_text_only(self):
        llm = LLM("models/fake", harness_name="search")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["<search>cats</search>", "answer"])
        llm.harness = SearchHarness(llm.engine)
        llm.harness._search = _SuccessfulSearch("curated cat result")
        session = llm.session([{"role": "user", "content": "Find cats"}])

        chunks = [chunk async for chunk in llm.harness.run(session)]

        self.assertEqual("".join(chunks), "answer")

    async def test_session_rejects_non_append_only_finalization(self):
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["broken"], tokenizer=_FinalizingRewriteTokenizer())
        llm.harness = DefaultHarness(llm.engine)
        session = llm.session([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "append-only prompt rendering"):
            await session.chat()

    async def test_finalize_assistant_requires_a_prepared_turn(self):
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = _ScriptedEngine(["unused"])
        llm.harness = DefaultHarness(llm.engine)
        session = llm.session([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            session._finalize_assistant("oops", [])

    async def test_session_chat_timeout_allows_long_responses_that_keep_progress(self):
        engine = _TokenDelayScriptedEngine(
            ["hello"],
            token_delays=[0.02, 0.02, 0.02, 0.02, 0.02],
        )
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = engine
        llm.harness = DefaultHarness(engine)
        session = llm.session([{"role": "user", "content": "Say hi"}])

        start = asyncio.get_running_loop().time()
        result = await session.chat(timeout=0.05)
        elapsed = asyncio.get_running_loop().time() - start

        self.assertEqual(result, "hello")
        self.assertGreater(elapsed, 0.08)

    async def test_session_chat_timeout_restarts_after_mid_stream_stall(self):
        engine = _TokenDelayScriptedEngine(
            ["hello"],
            token_delays=[0.0, 0.08],
        )
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = engine
        llm.harness = DefaultHarness(engine)
        session = llm.session([{"role": "user", "content": "Say hi"}])
        restart_calls: list[tuple] = []

        async def fake_swap_engine(
            model_dir: str,
            adapter_dir=None,
            harness_name=None,
            search_provider=None,
            num_threads=None,
            lora_quant=None,
            unembed_quant=None,
        ):
            restart_calls.append(
                (
                    model_dir,
                    adapter_dir,
                    harness_name,
                    search_provider,
                    num_threads,
                    lora_quant,
                    unembed_quant,
                )
            )
            recovered = _ScriptedEngine(["recovered"])
            llm.engine = recovered
            llm.harness = DefaultHarness(recovered)
            llm.state = ServerState.RUNNING
            llm._session_generation += 1
            return SimpleNamespace(status="success", model="fake", recompiled=False, message="")

        llm._swap_engine = fake_swap_engine

        with self.assertRaisesRegex(TimeoutError, "LLM chat timed out"):
            await session.chat(timeout=0.02)

        self.assertEqual(
            restart_calls,
            [("models/fake", None, "default", "ddgs", 0, None, None)],
        )
        with self.assertRaisesRegex(RuntimeError, "ChatSession is stale"):
            _ = session.messages
        self.assertEqual(
            await llm.session([{"role": "user", "content": "Try again"}]).chat(),
            "recovered",
        )


class LLMChatApiTests(unittest.IsolatedAsyncioTestCase):
    def _make_llm(self, response: str) -> LLM:
        engine = _ScriptedEngine([response])
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = engine
        llm.harness = DefaultHarness(engine)
        return llm

    async def test_stream_chat_emits_done_event_and_preserves_input_messages(self):
        llm = self._make_llm("ok")
        messages = [{"role": "user", "content": "hello"}]
        prompt_tokens = llm.session(messages).prompt_tokens

        events = [event async for event in llm.stream_chat(messages, max_tokens=8)]

        self.assertEqual(messages, [{"role": "user", "content": "hello"}])
        self.assertEqual([event.type for event in events], ["token", "token", "final_text", "done"])
        done = events[-1]
        self.assertIsInstance(done, ChatDoneEvent)
        self.assertEqual(done.text, "ok")
        self.assertEqual(done.usage.prompt_tokens, prompt_tokens)
        self.assertEqual(done.usage.completion_tokens, 2)
        self.assertEqual(done.usage.total_tokens, prompt_tokens + 2)
        self.assertEqual(done.usage.cached_tokens, 3)

    async def test_chat_returns_final_text(self):
        llm = self._make_llm("hello")

        result = await llm.chat([{"role": "user", "content": "Say hi"}], max_tokens=8)

        self.assertEqual(result, "hello")

    async def test_collect_chat_returns_usage(self):
        llm = self._make_llm("hello")

        text, usage = await llm._collect_chat(
            [{"role": "user", "content": "Say hi"}],
            max_tokens=8,
        )

        self.assertEqual(text, "hello")
        self.assertEqual(usage.completion_tokens, 5)

    async def test_chat_supports_timeout(self):
        engine = _SlowScriptedEngine(["hello"])
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = engine
        llm.harness = DefaultHarness(engine)
        restart_calls: list[tuple] = []

        async def fake_swap_engine(
            model_dir: str,
            adapter_dir=None,
            harness_name=None,
            search_provider=None,
            num_threads=None,
            lora_quant=None,
            unembed_quant=None,
        ):
            restart_calls.append(
                (
                    model_dir,
                    adapter_dir,
                    harness_name,
                    search_provider,
                    num_threads,
                    lora_quant,
                    unembed_quant,
                )
            )
            recovered = _ScriptedEngine(["recovered"])
            llm.engine = recovered
            llm.harness = DefaultHarness(recovered)
            llm.state = ServerState.RUNNING
            return SimpleNamespace(status="success", model="fake", recompiled=False, message="")

        llm._swap_engine = fake_swap_engine

        with self.assertRaisesRegex(TimeoutError, "LLM chat timed out"):
            await llm.chat(
                [{"role": "user", "content": "Say hi"}],
                max_tokens=8,
                timeout=0.001,
            )

        self.assertEqual(restart_calls, [("models/fake", None, "default", "ddgs", 0, None, None)])
        self.assertEqual(
            await llm.chat([{"role": "user", "content": "Try again"}], max_tokens=8),
            "recovered",
        )

    async def test_chat_timeout_surfaces_restart_failures(self):
        engine = _SlowScriptedEngine(["hello"])
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = engine
        llm.harness = DefaultHarness(engine)

        async def fake_swap_engine(*args, **kwargs):
            llm.state = ServerState.NO_MODEL
            return SimpleNamespace(
                status="error",
                model="fake",
                recompiled=False,
                message="boom",
            )

        llm._swap_engine = fake_swap_engine

        with self.assertRaisesRegex(
            RuntimeError,
            "LLM chat timed out and engine restart failed: boom",
        ):
            await llm.chat(
                [{"role": "user", "content": "Say hi"}],
                max_tokens=8,
                timeout=0.001,
            )

    async def test_llm_chat_is_one_turn_only(self):
        llm = self._make_llm("ignored")

        with self.assertRaisesRegex(ValueError, "assistant reply"):
            await llm.chat(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "already answered"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
