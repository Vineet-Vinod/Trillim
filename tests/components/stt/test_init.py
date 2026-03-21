"""Tests for STT package exports."""

import unittest

from trillim.components.stt import STT
from trillim.components.stt import __all__ as stt_exports


class STTInitTests(unittest.TestCase):
    def test_package_exports_only_stt(self):
        self.assertEqual(stt_exports, ["STT"])
        self.assertIsNotNone(STT)
