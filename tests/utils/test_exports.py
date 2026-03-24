"""Tests for utility package exports."""

import unittest

import trillim.utils as utils
from trillim.utils import stable_id


class UtilityExportTests(unittest.TestCase):
    def test_utility_exports_are_available(self):
        self.assertIs(utils.stable_id, stable_id)
        self.assertTrue(callable(stable_id))
        self.assertFalse(hasattr(utils, "ManagedSubprocess"))
