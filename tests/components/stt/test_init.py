"""Tests for STT package exports."""

import unittest

from trillim.components.stt import STT, STTSession
from trillim.components.stt import __all__ as stt_exports


class STTInitTests(unittest.TestCase):
    def test_package_exports_stt_and_stt_session(self):
        self.assertEqual(stt_exports, ["STT", "STTSession"])
        self.assertIsNotNone(STT)
        self.assertIsNotNone(STTSession)
