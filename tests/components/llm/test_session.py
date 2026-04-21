"""Tests for ChatSession behavior."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
import unittest

from trillim import _model_store
from trillim.components.llm import ChatDoneEvent, ChatFinalTextEvent, ChatTokenEvent, ChatUsage
from trillim.components.llm._engine import EngineCrashedError, EngineProgressTimeoutError
from trillim.components.llm._limits import SESSION_TOKEN_LIMIT
from trillim.components.llm._session import _CHAT_SESSION_OWNER_TOKEN, _ChatSession
from trillim.components.llm.public import LLM
from trillim.errors import (
    ContextOverflowError,
    InvalidRequestError,
    ProgressTimeoutError,
    SessionBusyError,
    SessionClosedError,
    SessionExhaustedError,
    SessionStaleError,
)
from trillim.harnesses.search.provider import SearchAuthenticationError
from tests.components.llm.support import (
    crashed,
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

    async def test_collect_chat_uses_token_final_and_done_events(self):
        llm = self._make_llm()

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def stream_chat(self, **_kwargs):
                yield ChatTokenEvent(text="a")
                yield ChatFinalTextEvent(text="partial")
                yield object()
                yield ChatDoneEvent(text="done", usage=ChatUsage(1, 1, 2, 0))

        llm.open_session = lambda _messages: _Session()

        text, usage = await llm._collect_chat([{"role": "user", "content": "hi"}])

        self.assertEqual(text, "done")
        self.assertEqual(usage.total_tokens, 2)

    def test_private_chat_session_init_rejects_invalid_owner_and_missing_args(self):
        session = object.__new__(_ChatSession)
        session._active_task = None
        with self.assertRaisesRegex(TypeError, "use LLM.open_session"):
            _ChatSession.__init__(session, object(), (), _owner_token=None)

        session = object.__new__(_ChatSession)
        session._active_task = None
        with self.assertRaisesRegex(TypeError, "use LLM.open_session"):
            _ChatSession.__init__(session, None, None, _owner_token=_CHAT_SESSION_OWNER_TOKEN)

    async def test_session_is_single_consumer(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        await session._begin_consumer()
        with self.assertRaisesRegex(SessionBusyError, "active consumer"):
            session.add_user("again")
        session._consumer_active = False
        session._active_task = None
        session._terminated.set()
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

    async def test_session_add_system_and_cached_token_count_properties(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        session._cached_token_count = 12

        session.add_system("facts")

        self.assertEqual(session.messages[-1], {"role": "system", "content": "facts"})
        self.assertEqual(session.cached_token_count, 12)
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

    async def test_session_close_from_active_consumer_clears_busy_state_even_if_stream_close_fails(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        class _BrokenEventStream:
            async def aclose(self):
                raise RuntimeError("close boom")

        await session._begin_consumer()
        session._active_event_stream = _BrokenEventStream()

        try:
            with self.assertRaisesRegex(RuntimeError, "close boom"):
                await session.close()
            self.assertEqual(session.state, "closed")
            self.assertFalse(session._consumer_active)
            self.assertIsNone(session._active_task)
            self.assertTrue(session._terminated.is_set())
        finally:
            session._active_event_stream = None
            session._consumer_active = False
            session._active_task = None
            session._terminated.set()
            await llm.stop()

    async def test_reset_stream_consumer_sets_terminated_after_cleanup(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()

        class _EventStream:
            async def aclose(self):
                cleanup_started.set()
                await cleanup_release.wait()

        session._active_event_stream = _EventStream()
        session._consumer_active = True
        session._terminated.clear()
        reset_task = asyncio.create_task(session._reset_stream_consumer())
        session._active_task = reset_task
        await cleanup_started.wait()
        wait_task = asyncio.create_task(session._wait_for_termination())
        await asyncio.sleep(0)
        self.assertFalse(wait_task.done())
        self.assertFalse(session._terminated.is_set())
        self.assertTrue(session._consumer_active)
        with self.assertRaisesRegex(SessionBusyError, "active consumer"):
            session._prepare_stream_chat(max_tokens=8)

        cleanup_release.set()
        await reset_task
        await wait_task

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

    async def test_session_recovers_from_engine_crash_and_raises_public_error(self):
        llm = self._make_llm(failure=crashed())
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "Inference engine crashed: boom"):
            await session.chat(max_tokens=8)

        self.assertEqual(llm.state.value, "running")
        await llm.stop()

    async def test_session_validation_error_is_not_masked_by_busy_admission(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        active_session = llm.open_session([{"role": "user", "content": "hello"}])
        invalid_session = llm.open_session([{"role": "user", "content": "hello"}])
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_stream_events(*_args, **_kwargs):
            started.set()
            await release.wait()
            yield ChatTokenEvent(text="ok")

        llm._harness.stream_events = blocking_stream_events

        async def consume():
            async for _event in active_session.stream_chat(max_tokens=8):
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        with self.assertRaisesRegex(InvalidRequestError, "greater than or equal to 0"):
            await invalid_session.chat(max_tokens=-1)
        release.set()
        await task
        await llm.stop()

    async def test_session_context_overflow_during_atomic_start_resets_state(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        async def overflowing_stream_events(*_args, **_kwargs):
            raise ContextOverflowError(10, 8)
            if False:  # pragma: no cover
                yield None

        llm._harness.stream_events = overflowing_stream_events

        with self.assertRaises(ContextOverflowError):
            async for _event in session.stream_chat(max_tokens=8):
                pass
        self.assertEqual(session.state, "open")
        self.assertTrue(session._terminated.is_set())
        await llm.stop()

    async def test_session_search_authentication_error_after_first_event_restores_state(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        seen = []

        async def auth_failing_stream_events(*_args, **_kwargs):
            yield ChatTokenEvent(text="a")
            raise SearchAuthenticationError("search auth failed")

        llm._harness.stream_events = auth_failing_stream_events

        with self.assertRaisesRegex(RuntimeError, "search auth failed"):
            async for event in session.stream_chat(max_tokens=8):
                seen.append(event)
        self.assertEqual(seen, [ChatTokenEvent(text="a")])
        self.assertEqual(session.state, "open")
        self.assertTrue(session._terminated.is_set())
        await llm.stop()

    async def test_session_progress_timeout_after_first_event_recovers_public_error(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        seen = []

        async def timeout_after_token(*_args, **_kwargs):
            yield ChatTokenEvent(text="a")
            raise EngineProgressTimeoutError("boom")

        llm._harness.stream_events = timeout_after_token

        with self.assertRaisesRegex(ProgressTimeoutError, "boom"):
            async for event in session.stream_chat(max_tokens=8):
                seen.append(event)
        self.assertEqual(seen, [ChatTokenEvent(text="a")])
        self.assertEqual(llm.state.value, "running")
        self.assertEqual(session.state, "stale")
        self.assertTrue(session._terminated.is_set())
        await llm.stop()

    async def test_session_handles_empty_started_stream(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        async def empty_stream_events(*_args, **_kwargs):
            if False:  # pragma: no cover
                yield None

        llm._harness.stream_events = empty_stream_events

        events = [event async for event in session.stream_chat(max_tokens=8)]
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ChatDoneEvent)
        self.assertEqual(events[0].text, "")
        self.assertEqual(session.state, "open")
        await llm.stop()

    async def test_session_generic_start_failure_marks_failed_and_cleans_up(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        async def failing_stream_events(*_args, **_kwargs):
            raise RuntimeError("boom")
            if False:  # pragma: no cover
                yield None

        llm._harness.stream_events = failing_stream_events

        with self.assertRaisesRegex(RuntimeError, "boom"):
            async for _event in session.stream_chat(max_tokens=8):
                pass
        self.assertEqual(session.state, "failed")
        self.assertTrue(session._terminated.is_set())
        await llm.stop()

    async def test_session_chat_requires_a_done_event(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        async def final_only_stream(**_kwargs):
            yield ChatFinalTextEvent(text="partial")

        session.stream_chat = final_only_stream
        with self.assertRaisesRegex(RuntimeError, "without a done event"):
            await session.chat(max_tokens=8)
        await llm.stop()

    async def test_session_chat_ignores_unknown_events_before_done(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        async def odd_stream(**_kwargs):
            yield object()
            yield ChatDoneEvent(text="done", usage=ChatUsage(1, 1, 2, 0))

        session.stream_chat = odd_stream

        self.assertEqual(await session.chat(max_tokens=8), "done")
        await llm.stop()

    async def test_session_cancel_during_started_stream_marks_session_closed(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        started = asyncio.Event()

        async def hanging_after_token(*_args, **_kwargs):
            yield ChatTokenEvent(text="a")
            started.set()
            await asyncio.Event().wait()

        llm._harness.stream_events = hanging_after_token

        async def consume():
            async for _event in session.stream_chat(max_tokens=8):
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(session.state, "closed")
        self.assertTrue(session._terminated.is_set())
        await llm.stop()

    async def test_session_context_overflow_after_first_event_restores_state(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        seen = []

        async def overflowing_after_token(*_args, **_kwargs):
            yield ChatTokenEvent(text="a")
            raise ContextOverflowError(10, 8)

        llm._harness.stream_events = overflowing_after_token

        with self.assertRaises(ContextOverflowError):
            async for event in session.stream_chat(max_tokens=8):
                seen.append(event)
        self.assertEqual(seen, [ChatTokenEvent(text="a")])
        self.assertEqual(session.state, "open")
        await llm.stop()

    async def test_session_exhaustion_after_first_event_marks_exhausted(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        seen = []

        async def exhausted_after_token(*_args, **_kwargs):
            yield ChatTokenEvent(text="a")
            raise SessionExhaustedError("exhausted")

        llm._harness.stream_events = exhausted_after_token

        with self.assertRaisesRegex(SessionExhaustedError, "exhausted"):
            async for event in session.stream_chat(max_tokens=8):
                seen.append(event)
        self.assertEqual(seen, [ChatTokenEvent(text="a")])
        self.assertEqual(session.state, "exhausted")
        await llm.stop()

    async def test_session_engine_crash_after_first_event_marks_failed_and_recovers(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        seen = []

        async def crashing_after_token(*_args, **_kwargs):
            yield ChatTokenEvent(text="a")
            raise EngineCrashedError("boom")

        llm._harness.stream_events = crashing_after_token

        with self.assertRaisesRegex(RuntimeError, "boom"):
            async for event in session.stream_chat(max_tokens=8):
                seen.append(event)
        self.assertEqual(seen, [ChatTokenEvent(text="a")])
        self.assertEqual(llm.state.value, "running")
        self.assertEqual(session.state, "stale")
        await llm.stop()

    async def test_session_generic_failure_after_first_event_marks_failed(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        seen = []

        async def failing_after_token(*_args, **_kwargs):
            yield ChatTokenEvent(text="a")
            raise RuntimeError("boom")

        llm._harness.stream_events = failing_after_token

        with self.assertRaisesRegex(RuntimeError, "boom"):
            async for event in session.stream_chat(max_tokens=8):
                seen.append(event)
        self.assertEqual(seen, [ChatTokenEvent(text="a")])
        self.assertEqual(session.state, "failed")
        await llm.stop()

    async def test_stream_helpers_cover_passthrough_and_noop_failure_paths(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        done = ChatDoneEvent(text="done", usage=ChatUsage(1, 1, 2, 0))
        self.assertEqual(session._accumulate_stream_text("prefix", done), "prefix")
        session._mark_stream_cancelled()
        self.assertEqual(session.state, "closed")
        session._state = "closed"
        session._mark_stream_failed(Exception)
        self.assertEqual(session.state, "closed")
        await llm.stop()

    async def test_session_turn_startable_rejects_empty_and_assistant_terminated_histories(self):
        llm = self._make_llm()
        await llm.start()
        empty = llm.open_session()

        with self.assertRaisesRegex(ValueError, "has no messages"):
            empty._prepare_stream_chat(max_tokens=8)

        answered = llm.open_session([{"role": "user", "content": "hello"}])
        answered._messages.append({"role": "assistant", "content": "done"})
        with self.assertRaisesRegex(ValueError, "already has an assistant reply"):
            answered._prepare_stream_chat(max_tokens=8)
        await llm.stop()

    async def test_begin_consumer_rejects_duplicate_active_consumer(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        await session._begin_consumer()
        with self.assertRaisesRegex(SessionBusyError, "active consumer"):
            await session._begin_consumer()

        session._consumer_active = False
        session._active_task = None
        session._terminated.set()
        await llm.stop()

    async def test_begin_consumer_private_helper_raises_when_already_busy(self):
        session = object.__new__(_ChatSession)
        session._consumer_active = True
        session._active_task = None

        with self.assertRaisesRegex(SessionBusyError, "active consumer"):
            await session._begin_consumer()

    async def test_prepare_generation_marks_exhaustion_and_context_overflow(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        llm._runtime_model = make_runtime_model(Path("/tmp/context-small"))
        llm._runtime_model = llm._runtime_model.__class__(
            name=llm._runtime_model.name,
            path=llm._runtime_model.path,
            arch_type=llm._runtime_model.arch_type,
            activation=llm._runtime_model.activation,
            hidden_dim=llm._runtime_model.hidden_dim,
            intermediate_dim=llm._runtime_model.intermediate_dim,
            num_layers=llm._runtime_model.num_layers,
            num_heads=llm._runtime_model.num_heads,
            num_kv_heads=llm._runtime_model.num_kv_heads,
            vocab_size=llm._runtime_model.vocab_size,
            head_dim=llm._runtime_model.head_dim,
            max_position_embeddings=1,
            norm_eps=llm._runtime_model.norm_eps,
            rope_theta=llm._runtime_model.rope_theta,
            eos_tokens=llm._runtime_model.eos_tokens,
            has_qkv_bias=llm._runtime_model.has_qkv_bias,
            tie_word_embeddings=llm._runtime_model.tie_word_embeddings,
            has_attn_sub_norm=llm._runtime_model.has_attn_sub_norm,
            has_ffn_sub_norm=llm._runtime_model.has_ffn_sub_norm,
        )
        with self.assertRaises(ContextOverflowError):
            session._prepare_generation()

        llm._runtime_model = llm._runtime_model.__class__(
            name=llm._runtime_model.name,
            path=llm._runtime_model.path,
            arch_type=llm._runtime_model.arch_type,
            activation=llm._runtime_model.activation,
            hidden_dim=llm._runtime_model.hidden_dim,
            intermediate_dim=llm._runtime_model.intermediate_dim,
            num_layers=llm._runtime_model.num_layers,
            num_heads=llm._runtime_model.num_heads,
            num_kv_heads=llm._runtime_model.num_kv_heads,
            vocab_size=llm._runtime_model.vocab_size,
            head_dim=llm._runtime_model.head_dim,
            max_position_embeddings=SESSION_TOKEN_LIMIT + 10,
            norm_eps=llm._runtime_model.norm_eps,
            rope_theta=llm._runtime_model.rope_theta,
            eos_tokens=llm._runtime_model.eos_tokens,
            has_qkv_bias=llm._runtime_model.has_qkv_bias,
            tie_word_embeddings=llm._runtime_model.tie_word_embeddings,
            has_attn_sub_norm=llm._runtime_model.has_attn_sub_norm,
            has_ffn_sub_norm=llm._runtime_model.has_ffn_sub_norm,
        )
        llm._tokenizer.encode = lambda *_args, **_kwargs: list(range(SESSION_TOKEN_LIMIT + 1))
        with self.assertRaisesRegex(SessionExhaustedError, "token lifetime limit"):
            session._prepare_generation()

        self.assertEqual(session.state, "exhausted")
        await llm.stop()

    async def test_render_prompt_and_termination_helpers_cover_empty_and_closed_paths(self):
        llm = self._make_llm()
        await llm.start()
        session = llm.open_session()

        self.assertEqual(session._render_prompt(add_generation_prompt=False), "")
        self.assertEqual(session._render_prompt(add_generation_prompt=True), "assistant: ")

        cancelled: list[str] = []

        class _Task:
            def done(self) -> bool:
                return False

            def cancel(self) -> None:
                cancelled.append("cancelled")

        session._active_task = _Task()
        session._mark_owner_stopped()
        self.assertEqual(cancelled, ["cancelled"])

        session._active_task = asyncio.current_task()
        await session._wait_for_termination()
        session._active_task = None

        session._state = "closed"
        done = session._finish_stream_success("done")
        self.assertIsInstance(done, ChatDoneEvent)
        self.assertEqual(session.state, "closed")
        await llm.stop()

    def test_chat_session_destructor_swallows_task_cleanup_errors(self):
        class _BrokenTask:
            def done(self) -> bool:
                raise RuntimeError("done boom")

            def cancel(self) -> None:
                raise RuntimeError("cancel boom")

        session = object.__new__(_ChatSession)
        session._active_task = _BrokenTask()

        session.__del__()

    def test_chat_session_destructor_cancels_live_tasks(self):
        class _Task:
            def __init__(self) -> None:
                self.cancelled = False

            def done(self) -> bool:
                return False

            def cancel(self) -> None:
                self.cancelled = True

        task = _Task()
        session = object.__new__(_ChatSession)
        session._active_task = task

        session.__del__()

        self.assertTrue(task.cancelled)

    async def test_chat_session_destructor_cancels_real_asyncio_tasks(self):
        async def sleeper() -> None:
            await asyncio.sleep(10)

        task = asyncio.create_task(sleeper())
        session = object.__new__(_ChatSession)
        session._active_task = task

        session.__del__()
        await asyncio.sleep(0)

        self.assertTrue(task.cancelled())

    async def test_prepared_stream_rechecks_closed_session_before_start(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        sampling = session._prepare_stream_chat(max_tokens=8)

        await session.close()

        with self.assertRaisesRegex(SessionClosedError, "closed"):
            async for _event in session._stream_chat_prepared(sampling):
                pass
        self.assertEqual(session.state, "closed")
        await llm.stop()

    async def test_prepared_stream_rechecks_stale_session_before_start(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        sampling = session._prepare_stream_chat(max_tokens=8)

        session._mark_stale()

        with self.assertRaisesRegex(SessionStaleError, "stale"):
            async for _event in session._stream_chat_prepared(sampling):
                pass
        self.assertEqual(session.state, "stale")
        await llm.stop()

    async def test_prepared_stream_rechecks_owner_stopped_session_before_start(self):
        llm = self._make_llm(responses=["ok"])
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])
        sampling = session._prepare_stream_chat(max_tokens=8)

        session._mark_owner_stopped()

        with self.assertRaisesRegex(SessionClosedError, "owner has stopped"):
            async for _event in session._stream_chat_prepared(sampling):
                pass
        self.assertEqual(session.state, "owner_stopped")
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
