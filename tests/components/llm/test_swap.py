"""Tests for LLM hot swap helpers."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import unittest

from trillim import _model_store
from trillim.components.llm._config import LLMState
from trillim.components.llm._swap import restart_model
from trillim.components.llm.public import LLM
from trillim.harnesses.search._harness import _SearchHarness
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
)


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
