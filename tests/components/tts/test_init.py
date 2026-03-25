"""Tests for TTS package exports."""

import unittest

from trillim.components.tts import TTS, TTSSession


class TTSInitTests(unittest.TestCase):
    def test_package_exports(self):
        self.assertIsNotNone(TTS)
        self.assertIsNotNone(TTSSession)

