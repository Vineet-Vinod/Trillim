"""Tests for the search harness."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from trillim import _model_store
from trillim.components.llm import ChatDoneEvent
from trillim.components.llm.public import LLM
from trillim.errors import SessionExhaustedError
from trillim.harnesses.search.provider import (
    FALLBACK_SEARCH_FAILURE_MESSAGE,
    SearchAuthenticationError,
    SearchError,
)
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
)


class _SuccessfulSearch:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[str] = []

    async def search(self, query: str) -> str:
        self.calls.append(query)
        return self.content


class _FailingSearch:
    async def search(self, query: str) -> str:
        raise SearchError(query)


class _AuthFailingSearch:
    async def search(self, query: str) -> str:
        raise SearchAuthenticationError("Brave search failed: wrong SEARCH_API_KEY")


class SearchHarnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patched_model_store())
        _model_store.store_path_for_id("Trillim/fake").mkdir(parents=True, exist_ok=True)

    def _make_llm(self, *, responses, search_token_budget: int = 32) -> LLM:
        return LLM(
            "Trillim/fake",
            harness_name="search",
            search_provider="ddgs",
            search_token_budget=search_token_budget,
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=responses),
        )

    async def test_search_harness_appends_search_history_and_reports_final_usage(self):
        llm = self._make_llm(responses=["<search>cats</search>", "answer"])
        await llm.start()
        llm._harness._search = _SuccessfulSearch("curated cat results")

        async with llm.open_session([{"role": "user", "content": "Find cats"}]) as session:
            events = [event async for event in session.stream_chat(max_tokens=8)]

        done = events[-1]
        self.assertIsInstance(done, ChatDoneEvent)
        self.assertEqual(
            session.messages,
            (
                {"role": "user", "content": "Find cats"},
                {"role": "assistant", "content": "<search>cats</search>"},
                {"role": "search", "content": "curated cat results"},
                {"role": "assistant", "content": "answer"},
            ),
        )
        self.assertEqual(
            done.usage.completion_tokens,
            len("answer"),
        )
        self.assertEqual(
            done.usage.prompt_tokens + done.usage.completion_tokens,
            llm._engine.cached_token_count,
        )
        self.assertEqual(done.usage.total_tokens, llm._engine.cached_token_count)
        self.assertEqual(done.usage.cached_tokens, llm._engine.last_cache_hit)
        await llm.stop()

    async def test_search_harness_uses_fallback_message_when_search_fails(self):
        llm = self._make_llm(responses=["<search>cats</search>", "answer"])
        await llm.start()
        llm._harness._search = _FailingSearch()

        async with llm.open_session([{"role": "user", "content": "Find cats"}]) as session:
            result = await session.chat(max_tokens=8)

        self.assertEqual(result, "answer")
        self.assertEqual(session.messages[2]["role"], "search")
        self.assertEqual(session.messages[2]["content"], FALLBACK_SEARCH_FAILURE_MESSAGE)
        await llm.stop()

    async def test_search_harness_auth_failures_leave_session_reusable(self):
        llm = self._make_llm(responses=["<search>cats</search>", "recovered"])
        await llm.start()
        llm._harness._search = _AuthFailingSearch()
        session = llm.open_session([{"role": "user", "content": "Find cats"}])

        with self.assertRaisesRegex(RuntimeError, "wrong SEARCH_API_KEY"):
            await session.chat(max_tokens=8)

        self.assertEqual(session.state, "open")
        self.assertEqual(
            session.messages,
            ({"role": "user", "content": "Find cats"},),
        )
        self.assertEqual(await session.chat(max_tokens=8), "recovered")
        await llm.stop()

    async def test_search_harness_trims_search_content_to_token_budget(self):
        llm = self._make_llm(
            responses=["<search>cats</search>", "answer"],
            search_token_budget=4,
        )
        await llm.start()
        llm._harness._search = _SuccessfulSearch("abcdefghijklmnop")

        async with llm.open_session([{"role": "user", "content": "Find cats"}]) as session:
            await session.chat(max_tokens=8)

        self.assertLessEqual(
            len(llm._tokenizer.encode(session.messages[2]["content"], add_special_tokens=False)),
            4,
        )
        await llm.stop()

    async def test_search_harness_exhausts_when_follow_up_prompt_crosses_session_limit(self):
        llm = self._make_llm(responses=["<search>cats</search>", "unused"])
        await llm.start()
        llm._harness._search = _SuccessfulSearch("x" * 48)
        session = llm.open_session([{"role": "user", "content": "Find cats"}])

        with patch("trillim.components.llm._session.SESSION_TOKEN_LIMIT", 80):
            with self.assertRaises(SessionExhaustedError):
                await session.chat(max_tokens=8)

        self.assertEqual(session.state, "exhausted")
        self.assertEqual(
            session.messages,
            ({"role": "user", "content": "Find cats"},),
        )
        await llm.stop()

    async def test_search_harness_without_tags_returns_buffered_text(self):
        llm = self._make_llm(responses=["hello"])
        await llm.start()

        async with llm.open_session([{"role": "user", "content": "Say hi"}]) as session:
            events = [event async for event in session.stream_chat(max_tokens=8)]

        self.assertEqual([event.type for event in events], ["token", "final_text", "done"])
        self.assertEqual(events[-1].text, "hello")
        await llm.stop()

    async def test_search_harness_uses_engine_reported_usage(self):
        llm = self._make_llm(responses=["hello"])
        await llm.start()

        async with llm.open_session([{"role": "user", "content": "Say hi"}]) as session:
            prompt_tokens = len(session._prepare_generation())
            llm._engine.kv_positions = [prompt_tokens + 1]
            events = [event async for event in session.stream_chat(max_tokens=8)]

        done = events[-1]
        self.assertEqual(done.usage.prompt_tokens, prompt_tokens)
        self.assertEqual(done.usage.completion_tokens, 1)
        self.assertEqual(done.usage.total_tokens, prompt_tokens + 1)
        await llm.stop()
