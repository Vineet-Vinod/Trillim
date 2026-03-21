"""Tests for stable identifier helpers."""

import unittest

from trillim.utils.ids import stable_id


class StableIdTests(unittest.TestCase):
    def test_stable_id_is_deterministic(self):
        self.assertEqual(stable_id("voice", "demo"), stable_id("voice", "demo"))
        self.assertNotEqual(stable_id("voice", "demo"), stable_id("voice", "other"))

    def test_stable_id_validates_arguments(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            stable_id("-", "demo")
        with self.assertRaisesRegex(ValueError, "digest_size"):
            stable_id("voice", "demo", digest_size=2)

