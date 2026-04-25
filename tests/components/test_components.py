"""Tests for skeletal component packages."""

from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path

from fastapi import APIRouter

from trillim import _model_store
from trillim.components import Component
from trillim.components.llm import LLM
from trillim.components.stt import STT
from trillim.components.tts import TTS
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
)
from tests.components.tts.support import patched_tts_environment


class ComponentSkeletonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patched_model_store())
        _model_store.store_path_for_id("Trillim/fake").mkdir(parents=True, exist_ok=True)

    async def test_component_packages_are_valid_components(self):
        llm = LLM(
            "Trillim/fake",
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        for component in (llm, STT(), TTS()):
            self.assertIsInstance(component, Component)
            self.assertIsInstance(component.router(), APIRouter)
        await llm.start()
        await llm.stop()
        stt = STT()
        await stt.start()
        await stt.stop()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patched_tts_environment(Path(temp_dir) / "voices"):
                tts = TTS()
                await tts.start()
                await tts.stop()

    async def test_component_names_match_expected_runtime_names(self):
        llm = LLM(
            "Trillim/fake",
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        self.assertEqual(llm.component_name, "llm")
        self.assertEqual(STT().component_name, "stt")
        self.assertEqual(TTS().component_name, "tts")

    async def test_base_component_defaults_are_safe_noops(self):
        component = Component()

        self.assertEqual(component.component_name, "component")
        self.assertIsInstance(component.router(), APIRouter)
        self.assertIsNone(await component.start())
        self.assertIsNone(await component.stop())
