"""Tests for public LLM package exports."""

import unittest

from trillim.components.llm import (
    ChatDoneEvent,
    ChatEvent,
    ChatFinalTextEvent,
    ChatSession,
    ChatTokenEvent,
    ChatUsage,
    LLM,
    ModelInfo,
)


class LLMExportTests(unittest.TestCase):
    def test_llm_package_exports_are_available(self):
        self.assertIsNotNone(LLM)
        self.assertIsNotNone(ChatSession)
        self.assertIsNotNone(ChatUsage)
        self.assertIsNotNone(ChatTokenEvent)
        self.assertIsNotNone(ChatFinalTextEvent)
        self.assertIsNotNone(ChatDoneEvent)
        self.assertIsNotNone(ChatEvent)
        self.assertIsNotNone(ModelInfo)
