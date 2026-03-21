"""Tests for skeletal component packages."""

import unittest

from fastapi import APIRouter

from trillim.components import Component
from trillim.components.llm import LLM
from trillim.components.stt import STT
from trillim.components.tts import TTS


class ComponentSkeletonTests(unittest.IsolatedAsyncioTestCase):
    async def test_placeholder_components_are_valid_components(self):
        for component in (LLM(), STT(), TTS()):
            self.assertIsInstance(component, Component)
            self.assertIsInstance(component.router(), APIRouter)
            await component.start()
            await component.stop()

    async def test_placeholder_component_names_match_expected_runtime_names(self):
        self.assertEqual(LLM().component_name, "llm")
        self.assertEqual(STT().component_name, "stt")
        self.assertEqual(TTS().component_name, "tts")

