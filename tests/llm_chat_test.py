# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for structured chat events and the public LLM chat API."""

import asyncio
from types import SimpleNamespace
import unittest

from trillim.events import ChatDoneEvent, ChatFinalTextEvent, ChatSearchResultEvent
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


class _ScriptedEngine:
    def __init__(self, responses: list[str], *, max_context_tokens: int = 4096):
        self._responses = list(responses)
        self.tokenizer = _FakeTokenizer()
        self.arch_config = SimpleNamespace(max_position_embeddings=max_context_tokens)
        self._cached_prompt_str = ""
        self._last_cache_hit = 3

    async def generate(self, **_):
        response = self._responses.pop(0)
        for ch in response:
            yield ord(ch)


class _SlowScriptedEngine(_ScriptedEngine):
    async def generate(self, **kwargs):
        await asyncio.sleep(0.05)
        async for token in super().generate(**kwargs):
            yield token


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


class HarnessEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_harness_emits_token_and_final_text_events(self):
        harness = DefaultHarness(_ScriptedEngine(["hi"]))
        messages = [{"role": "user", "content": "hello"}]

        events = [event async for event in harness.stream_events(messages)]

        self.assertEqual([event.type for event in events], ["token", "token", "final_text"])
        self.assertEqual("".join(event.text for event in events[:-1]), "hi")
        self.assertIsInstance(events[-1], ChatFinalTextEvent)
        self.assertEqual(events[-1].text, "hi")
        self.assertEqual(messages[-1], {"role": "assistant", "content": "hi"})

    async def test_search_harness_emits_structured_search_events(self):
        harness = SearchHarness(_ScriptedEngine(["<search>cats</search>", "answer"]))
        harness._search = _SuccessfulSearch("curated cat result")
        messages = [{"role": "user", "content": "Find cats"}]

        events = [event async for event in harness.stream_events(messages)]

        self.assertEqual(events[0].type, "search_started")
        self.assertEqual(events[0].query, "cats")
        self.assertIsInstance(events[1], ChatSearchResultEvent)
        self.assertTrue(events[1].available)
        self.assertEqual(events[1].content, "curated cat result")
        self.assertEqual("".join(event.text for event in events if event.type == "token"), "answer")
        self.assertEqual(events[-1].type, "final_text")
        self.assertEqual(events[-1].text, "answer")

    async def test_search_harness_marks_unavailable_searches(self):
        harness = SearchHarness(_ScriptedEngine(["<search>cats</search>", "fallback"]))
        harness._search = _UnavailableSearch()

        events = [event async for event in harness.stream_events([{"role": "user", "content": "Find cats"}])]

        self.assertEqual(events[0].type, "search_started")
        self.assertIsInstance(events[1], ChatSearchResultEvent)
        self.assertFalse(events[1].available)
        self.assertIn("Search unavailable", events[1].content)
        self.assertEqual(events[-1].text, "fallback")

    async def test_harness_run_remains_plain_text_only(self):
        harness = SearchHarness(_ScriptedEngine(["<search>cats</search>", "answer"]))
        harness._search = _SuccessfulSearch("curated cat result")

        chunks = [chunk async for chunk in harness.run([{"role": "user", "content": "Find cats"}])]

        self.assertEqual("".join(chunks), "answer")


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
        prompt_tokens = llm.count_tokens(messages)

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

    async def test_chat_supports_timeout(self):
        engine = _SlowScriptedEngine(["hello"])
        llm = LLM("models/fake")
        llm.state = ServerState.RUNNING
        llm.engine = engine
        llm.harness = DefaultHarness(engine)

        with self.assertRaisesRegex(TimeoutError, "LLM chat timed out"):
            await llm.chat(
                [{"role": "user", "content": "Say hi"}],
                max_tokens=8,
                timeout=0.001,
            )


if __name__ == "__main__":
    unittest.main()
