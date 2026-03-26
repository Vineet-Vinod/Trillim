"""Tests for LLM hot swap helpers."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest

from trillim import _model_store
from trillim.components.llm._config import LLMState
from trillim.components.llm._swap import _best_effort_stop, _wait_for_idle_or_cancel, restart_model
from trillim.components.llm.public import LLM
from trillim.harnesses.search._harness import _SearchHarness
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
)


class _SwapEngine:
    def __init__(self, *, start_error: Exception | None = None, stop_error: Exception | None = None):
        self.start_error = start_error
        self.stop_error = stop_error
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


class _RuntimeFiles:
    def __init__(self) -> None:
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class _Admission:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[float] = []
        self.finish_swapping_calls = 0

    async def wait_for_idle(self, *, timeout: float) -> None:
        self.calls.append(timeout)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome

    async def finish_swapping(self) -> None:
        self.finish_swapping_calls += 1


class SwapTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patched_model_store())

    def _ensure_store_dir(self, store_id: str) -> None:
        _model_store.store_path_for_id(store_id).mkdir(parents=True, exist_ok=True)

    async def test_swap_model_stops_old_engine_before_starting_new_one(self):
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        lifecycle_log: list[str] = []
        model_iter = iter(models)
        factory = FakeEngineFactory(responses=["one"], lifecycle_log=lifecycle_log)
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        await llm.start()

        await llm.swap_model("Trillim/two")

        self.assertEqual(
            lifecycle_log,
            ["start:model-one", "stop:model-one", "start:model-two"],
        )
        await llm.stop()

    async def test_swap_model_replaces_runtime_and_stales_sessions(self):
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        model_iter = iter(models)
        factory = FakeEngineFactory(responses=["one"])
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        info = await llm.swap_model("Trillim/two")

        self.assertEqual(info.name, "model-two")
        self.assertEqual(session.state, "stale")
        self.assertEqual(len(factory.instances), 2)
        self.assertEqual(factory.instances[0].stop_calls, 1)
        await llm.stop()

    async def test_swap_model_enters_server_error_when_new_start_fails(self):
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        model_iter = iter(models)
        factory = FakeEngineFactory(responses=["one"])
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        await llm.start()
        llm._engine_factory = FakeEngineFactory(start_error=RuntimeError("boom"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await llm.swap_model("Trillim/two")

        self.assertEqual(llm.state, LLMState.SERVER_ERROR)
        self.assertIsNone(llm.model_name)

    async def test_swap_model_can_recover_from_server_error_with_a_valid_model(self):
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
            make_runtime_model(Path("/tmp/model-three"), name="model-three"),
        ]
        model_iter = iter(models)
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        self._ensure_store_dir("Trillim/three")
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["one"]),
        )
        await llm.start()
        llm._engine_factory = FakeEngineFactory(start_error=RuntimeError("boom"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await llm.swap_model("Trillim/two")

        self.assertEqual(llm.state, LLMState.SERVER_ERROR)

        llm._engine_factory = FakeEngineFactory(responses=["three"])
        info = await llm.swap_model("Trillim/three")

        self.assertEqual(info.state, LLMState.RUNNING)
        self.assertEqual(info.name, "model-three")
        self.assertEqual(llm.model_name, "model-three")
        await llm.stop()

    async def test_swap_model_updates_harness_provider_and_budget(self):
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        model_iter = iter(models)
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["one"]),
        )
        await llm.start()

        await llm.swap_model(
            "Trillim/two",
            harness_name="search",
            search_provider="BRAVE_SEARCH",
            search_token_budget=2048,
        )

        self.assertIsInstance(llm._harness, _SearchHarness)
        self.assertEqual(llm._configured_harness_name, "search")
        self.assertEqual(llm._configured_search_provider, "brave")
        self.assertEqual(llm._configured_search_token_budget, 2048)
        self.assertEqual(llm._runtime_search_token_budget, 1024)
        await llm.stop()

    async def test_swap_model_preflight_failure_keeps_existing_runtime_running(self):
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        model_iter = iter(models)
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["one", "still-one"]),
        )
        await llm.start()

        with self.assertRaisesRegex(ValueError, "search_token_budget must be at least 1"):
            await llm.swap_model("Trillim/two", search_token_budget=0)

        self.assertEqual(llm.state, LLMState.RUNNING)
        self.assertEqual(llm.model_name, "model-one")
        self.assertEqual(
            await llm.chat([{"role": "user", "content": "hi"}]),
            "one",
        )
        self.assertEqual(
            await llm.chat([{"role": "user", "content": "again"}]),
            "still-one",
        )
        await llm.stop()

    async def test_restart_model_preserves_search_runtime_options(self):
        self._ensure_store_dir("Trillim/one")
        llm = LLM(
            "Trillim/one",
            harness_name="search",
            search_provider="BRAVE_SEARCH",
            search_token_budget=2048,
            _model_validator=lambda _: make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["one"]),
        )
        await llm.start()

        await restart_model(llm)

        self.assertIsInstance(llm._harness, _SearchHarness)
        self.assertEqual(llm._configured_harness_name, "search")
        self.assertEqual(llm._configured_search_provider, "brave")
        self.assertEqual(llm._runtime_search_token_budget, 1024)
        await llm.stop()

    async def test_restart_model_does_not_restore_runtime_when_stop_wins_after_handoff(self):
        self._ensure_store_dir("Trillim/one")
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        model_iter = iter(models)
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["one"]),
        )
        await llm.start()
        entered = asyncio.Event()
        release = asyncio.Event()
        original_finish_swapping = llm._admission.finish_swapping

        async def blocked_finish_swapping() -> None:
            entered.set()
            await release.wait()
            await original_finish_swapping()

        llm._admission.finish_swapping = blocked_finish_swapping
        restart_task = asyncio.create_task(restart_model(llm))
        await entered.wait()
        stop_task = asyncio.create_task(llm.stop())

        try:
            await asyncio.sleep(0.05)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            release.set()
            restart_result, stop_result = await asyncio.gather(
                restart_task,
                stop_task,
                return_exceptions=True,
            )
            self.assertIsNone(restart_result)
            self.assertIsNone(stop_result)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            self.assertIsNone(llm.model_name)
        finally:
            release.set()
            await asyncio.gather(restart_task, stop_task, return_exceptions=True)
            if llm.state == LLMState.RUNNING:
                await llm.stop()

    async def test_restart_model_sets_server_error_when_no_runtime_model_exists(self):
        calls: list[str] = []
        llm = SimpleNamespace(
            _swap_lock=asyncio.Lock(),
            _runtime_model=None,
            _set_server_error=lambda: calls.append("server_error"),
        )

        await restart_model(llm)

        self.assertEqual(calls, ["server_error"])

    async def test_restart_model_cleans_up_failed_rebuilds(self):
        calls: list[str] = []
        old_engine = _SwapEngine()
        failed_engine = _SwapEngine(start_error=RuntimeError("boom"))
        runtime_files = _RuntimeFiles()
        built_runtime = SimpleNamespace(
            engine=failed_engine,
            runtime_files=runtime_files,
        )
        llm = SimpleNamespace(
            _swap_lock=asyncio.Lock(),
            _runtime_model=object(),
            _configured_init_config="init",
            _configured_harness_name="search",
            _configured_search_provider="brave",
            _configured_search_token_budget=1024,
            _engine=old_engine,
            _state=LLMState.RUNNING,
            _admission=_Admission(),
            _build_runtime=lambda *args, **kwargs: built_runtime,
            _begin_swap=lambda: asyncio.sleep(0),
            _clear_runtime=lambda: calls.append("clear_runtime"),
            _bind_runtime=lambda runtime: calls.append(f"bind:{runtime!r}"),
            _set_server_error=lambda: calls.append("server_error"),
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await restart_model(llm)

        self.assertTrue(runtime_files.cleaned)
        self.assertEqual(old_engine.stop_calls, 2)
        self.assertEqual(failed_engine.stop_calls, 1)
        self.assertEqual(calls, ["clear_runtime", "server_error"])

    async def test_wait_for_idle_or_cancel_handles_success_cancellation_and_failure(self):
        llm = SimpleNamespace(
            _admission=_Admission([None]),
            _cancel_active_sessions=lambda: asyncio.sleep(0),
            _engine=None,
        )
        await _wait_for_idle_or_cancel(llm)
        self.assertEqual(len(llm._admission.calls), 1)

        cancel_calls: list[str] = []
        llm = SimpleNamespace(
            _admission=_Admission([TimeoutError(), None]),
            _cancel_active_sessions=lambda: cancel_calls.append("cancel") or asyncio.sleep(0),
            _engine=None,
        )
        await _wait_for_idle_or_cancel(llm)
        self.assertEqual(cancel_calls, ["cancel"])

        engine = _SwapEngine()
        llm = SimpleNamespace(
            _admission=_Admission([TimeoutError(), TimeoutError(), None]),
            _cancel_active_sessions=lambda: cancel_calls.append("cancel-again") or asyncio.sleep(0),
            _engine=engine,
        )
        await _wait_for_idle_or_cancel(llm)
        self.assertEqual(engine.stop_calls, 1)

        failing_engine = _SwapEngine()
        llm = SimpleNamespace(
            _admission=_Admission([TimeoutError(), TimeoutError(), TimeoutError()]),
            _cancel_active_sessions=lambda: asyncio.sleep(0),
            _engine=failing_engine,
        )
        with self.assertRaisesRegex(RuntimeError, "failed to halt active generations"):
            await _wait_for_idle_or_cancel(llm)
        self.assertEqual(failing_engine.stop_calls, 1)

    async def test_best_effort_stop_handles_none_and_engine_errors(self):
        await _best_effort_stop(None)
        await _best_effort_stop(_SwapEngine(stop_error=RuntimeError("boom")))
