"""Tests for top-level package exports."""

import unittest

import trillim
from trillim import LLM, Runtime, STT, Server, TTS


class PackageExportTests(unittest.TestCase):
    def test_top_level_exports_exist(self):
        self.assertIs(trillim.LLM, LLM)
        self.assertIs(trillim.STT, STT)
        self.assertIs(trillim.TTS, TTS)
        self.assertIs(trillim.Runtime, Runtime)
        self.assertIs(trillim.Server, Server)

