# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for shared sampling parameter schemas."""

import unittest

from pydantic import ValidationError

from trillim._sampling import EngineSamplingParams, HttpSamplingParams, first_validation_error


class SamplingSchemaTests(unittest.TestCase):
    def test_http_sampling_params_validate_shared_fields(self):
        with self.assertRaisesRegex(ValidationError, "temperature must be >= 0"):
            HttpSamplingParams(temperature=-0.1)

        with self.assertRaisesRegex(ValidationError, "top_p must be in \\(0, 1\\]"):
            HttpSamplingParams(top_p=0)

        with self.assertRaisesRegex(ValidationError, "top_k must be >= 1"):
            HttpSamplingParams(top_k=0)

        with self.assertRaisesRegex(ValidationError, "repetition_penalty must be >= 0"):
            HttpSamplingParams(repetition_penalty=-1)

    def test_http_sampling_params_reject_zero_max_tokens(self):
        with self.assertRaisesRegex(ValidationError, "max_tokens must be >= 1"):
            HttpSamplingParams(max_tokens=0)

    def test_engine_sampling_params_allow_zero_max_tokens(self):
        params = EngineSamplingParams(max_tokens=0, rep_penalty_lookback=64)

        self.assertEqual(params.max_tokens, 0)
        self.assertEqual(params.rep_penalty_lookback, 64)

    def test_engine_sampling_params_reject_negative_max_tokens(self):
        with self.assertRaisesRegex(ValidationError, "max_tokens must be >= 0"):
            EngineSamplingParams(max_tokens=-1)

    def test_first_validation_error_uses_first_message_and_fallback(self):
        with self.assertRaises(ValidationError) as ctx:
            HttpSamplingParams(top_k=0)

        self.assertEqual(first_validation_error(ctx.exception), "top_k must be >= 1")

        class _EmptyErrors:
            def errors(self, include_url=False):
                return []

            def __str__(self):
                return "fallback message"

        self.assertEqual(first_validation_error(_EmptyErrors()), "fallback message")

    def test_first_validation_error_preserves_non_prefixed_message(self):
        class _CustomErrors:
            def errors(self, include_url=False):
                del include_url
                return [{"msg": "already clean"}]

        self.assertEqual(first_validation_error(_CustomErrors()), "already clean")


if __name__ == "__main__":
    unittest.main()
