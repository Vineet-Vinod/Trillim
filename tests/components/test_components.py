"""Tests for skeletal component packages."""

import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import APIRouter

from trillim.components import Component
from trillim.components.llm import LLM
from trillim.components.stt import STT
from trillim.components.tts import TTS
from tests.components.llm.support import FakeEngineFactory, FakeTokenizer, make_runtime_model
from tests.components.stt.support import make_faster_whisper_stub


class ComponentSkeletonTests(unittest.IsolatedAsyncioTestCase):
    async def test_component_packages_are_valid_components(self):
        llm = LLM(
            "models/fake",
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        for component in (llm, STT(), TTS()):
            self.assertIsInstance(component, Component)
            self.assertIsInstance(component.router(), APIRouter)
        await llm.start()
        await llm.stop()
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}):
            stt = STT()
            await stt.start()
            await stt.stop()
        await TTS().start()
        await TTS().stop()

    async def test_component_names_match_expected_runtime_names(self):
        llm = LLM(
            "models/fake",
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        self.assertEqual(llm.component_name, "llm")
        self.assertEqual(STT().component_name, "stt")
        self.assertEqual(TTS().component_name, "tts")
