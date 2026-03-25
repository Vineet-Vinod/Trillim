"""Tests for ChatSession behavior."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
import unittest

from trillim import _model_store
from trillim.components.llm import ChatDoneEvent, ChatTokenEvent
from trillim.components.llm.public import LLM
from trillim.errors import (
    ProgressTimeoutError,
    SessionBusyError,
    SessionClosedError,
    SessionExhaustedError,
    SessionStaleError,
)
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
    progress_timeout,
)


class _StrictTemplateTokenizer(FakeTokenizer):
    chat_template = "strict"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        del tokenize
        for message in messages:
            if message["role"] not in {"system", "user", "assistant"}:
                raise AssertionError(f"unexpected role {message['role']}")
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class ChatSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patched_model_store())
        _model_store.store_path_for_id("Trillim/fake").mkdir(parents=True, exist_ok=True)
        _model_store.store_path_for_id("Trillim/other").mkdir(parents=True, exist_ok=True)

    def _make_llm(self, *, responses=None, kv_positions=None, failure=None):
        return LLM(
            "Trillim/fake",
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(
                responses=responses or ["ok"],
                kv_positions=kv_positions,
                failure=failure,
            ),
        )

    async def test_session_chat_updates_messages_and_usage(self):
        llm = self._make_llm(responses=["hello"])
        await llm.start()

        async with llm.open_session([{"role": "user", "content": "Say hi"}]) as session:
            events = [event async for event in session.stream_chat(max_tokens=8)]

        self.assertIsInstance(events[-1], ChatDoneEvent)
        self.assertEqual(session.messages[-1]["role"], "assistant")
        self.assertEqual(session.messages[-1]["content"], "hello")
        await llm.stop()

    async def test_session_is_single_consumer(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        await session._begin_consumer()
        with self.assertRaisesRegex(SessionBusyError, "active consumer"):
            session.add_user("again")
        session._consumer_active = False
        session._active_task = None
        await llm.stop()

    async def test_session_stales_after_swap_request(self):
        llm = self._make_llm(responses=["one", "two"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        await llm.swap_model("Trillim/other")

        with self.assertRaisesRegex(SessionStaleError, "stale"):
            session.add_user("again")
        await llm.stop()

    async def test_session_close_marks_session_closed(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        await session.close()

        with self.assertRaisesRegex(SessionClosedError, "closed"):
            session.add_user("x")
        await llm.stop()

    async def test_session_close_waits_for_active_consumer_cleanup(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def blocking_stream_events(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()
            if False:
                yield None

        llm._harness.stream_events = blocking_stream_events

        async def consume():
            async for _event in session.stream_chat(max_tokens=8):
                pass

        task = asyncio.create_task(consume())
        await started.wait()

        await session.close()

        self.assertTrue(task.done())
        self.assertTrue(cleaned.is_set())
        with self.assertRaises(asyncio.CancelledError):
            await task
        await llm.stop()

    async def test_session_close_from_active_consumer_waits_for_cleanup(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        cleaned = asyncio.Event()

        async def blocking_stream_events(*_args, **_kwargs):
            try:
                yield ChatTokenEvent(text="hello")
                await asyncio.Future()
            finally:
                cleaned.set()

        llm._harness.stream_events = blocking_stream_events

        seen = []
        async for event in session.stream_chat(max_tokens=8):
            seen.append(event)
            await session.close()
            self.assertTrue(cleaned.is_set())

        self.assertEqual(seen, [ChatTokenEvent(text="hello")])
        self.assertEqual(session.state, "closed")
        self.assertTrue(session._terminated.is_set())
        await llm.stop()

    async def test_session_exhausts_after_kv_position_limit(self):
        llm = self._make_llm(responses=["ok"], kv_positions=[256 * 1024])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        await session.chat(max_tokens=8)

        with self.assertRaises(SessionExhaustedError):
            session.add_user("again")
        await llm.stop()

    async def test_session_recovers_from_progress_timeout_and_raises_public_error(self):
        llm = self._make_llm(failure=progress_timeout())
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        with self.assertRaises(ProgressTimeoutError):
            await session.chat(max_tokens=8)

        self.assertEqual(llm.state.value, "running")
        await llm.stop()

    async def test_session_renders_search_messages_for_chat_templates(self):
        llm = LLM(
            "Trillim/fake",
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: _StrictTemplateTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        await llm.start()
        session = llm.open_session(
            [
                {"role": "user", "content": "hello"},
                {"role": "search", "content": "facts"},
            ]
        )

        prompt = session._render_prompt(add_generation_prompt=True)

        self.assertIn("<system>Search results:\nfacts</system>", prompt)
        await llm.stop()
