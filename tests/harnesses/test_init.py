"""Tests for harness package exports."""

import unittest

from trillim.harnesses import DefaultHarness, Harness


class HarnessExportTests(unittest.TestCase):
    def test_harness_exports_are_available(self):
        self.assertTrue(issubclass(DefaultHarness, Harness))
