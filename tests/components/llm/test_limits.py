"""Tests for LLM limits constants."""

import unittest

from trillim.components.llm import _limits


class LLMLimitTests(unittest.TestCase):
    def test_limits_are_bounded_and_positive(self):
        self.assertGreater(_limits.REQUEST_BODY_LIMIT_BYTES, 0)
        self.assertGreater(_limits.TOTAL_MESSAGE_TEXT_LIMIT_BYTES, 0)
        self.assertGreater(_limits.MAX_MESSAGES, 0)
        self.assertGreater(_limits.MAX_OUTPUT_TOKENS, _limits.DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(_limits.MAX_ACTIVE_GENERATIONS, 1)
