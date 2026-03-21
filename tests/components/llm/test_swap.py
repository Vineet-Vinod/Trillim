"""Tests for LLM hot swap helpers."""

from __future__ import annotations

from pathlib import Path
import unittest

from trillim.components.llm._config import LLMState
from trillim.components.llm.public import LLM
from tests.components.llm.support import FakeEngineFactory, FakeTokenizer, make_runtime_model


class SwapTests(unittest.IsolatedAsyncioTestCase):
    async def test_swap_model_stops_old_engine_before_starting_new_one(self):
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        lifecycle_log: list[str] = []
        model_iter = iter(models)
        factory = FakeEngineFactory(responses=["one"], lifecycle_log=lifecycle_log)
        llm = LLM(
            "models/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        await llm.start()

        await llm.swap_model("models/two")

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
        llm = LLM(
            "models/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        await llm.start()
        session = llm.open_session([{"role": "user", "content": "hello"}])

        info = await llm.swap_model("models/two")

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
        llm = LLM(
            "models/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        await llm.start()
        llm._engine_factory = FakeEngineFactory(start_error=RuntimeError("boom"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await llm.swap_model("models/two")

        self.assertEqual(llm.state, LLMState.SERVER_ERROR)
        self.assertIsNone(llm.model_name)
