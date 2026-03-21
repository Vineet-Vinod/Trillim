"""Tests for utility package exports."""

import unittest

from trillim.utils import ManagedSubprocess, stable_id


class UtilityExportTests(unittest.TestCase):
    def test_utility_exports_are_available(self):
        self.assertTrue(callable(stable_id))
        self.assertIsNotNone(ManagedSubprocess)

