"""Tests for LLM request validation."""

import unittest

from trillim.components.llm._validation import (
    validate_chat_request,
    validate_messages,
    validate_sampling_options,
    validate_swap_request,
)
from trillim.errors import InvalidRequestError


class ValidationTests(unittest.TestCase):
    def test_validate_chat_request_rejects_mismatched_models(self):
        with self.assertRaisesRegex(InvalidRequestError, "does not match"):
            validate_chat_request(
                {
                    "model": "other",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                active_model_name="active",
            )

    def test_validate_chat_request_rejects_assistant_last_message(self):
        with self.assertRaisesRegex(InvalidRequestError, "assistant reply"):
            validate_chat_request(
                {
                    "messages": [{"role": "assistant", "content": "done"}],
                },
                active_model_name="active",
            )

    def test_validate_messages_enforces_byte_budget(self):
        almost_limit = "x" * 262_144
        with self.assertRaisesRegex(InvalidRequestError, "total text budget"):
            validate_messages(
                [
                    {"role": "user", "content": almost_limit},
                    {"role": "user", "content": almost_limit},
                    {"role": "user", "content": almost_limit},
                    {"role": "user", "content": almost_limit},
                    {"role": "user", "content": "overflow"},
                ],
                require_user_turn=False,
                allow_empty=True,
            )

    def test_validate_sampling_options_and_swap_request(self):
        sampling = validate_sampling_options(max_tokens=32, top_k=10)
        swap = validate_swap_request(
            {
                "model_dir": "/tmp/model",
                "harness_name": "search",
                "search_provider": "BRAVE_SEARCH",
                "search_token_budget": 32,
            }
        )

        self.assertEqual(sampling.max_tokens, 32)
        self.assertEqual(swap.model_dir, "/tmp/model")
        self.assertEqual(swap.harness_name, "search")
        self.assertEqual(swap.search_provider, "BRAVE_SEARCH")
        self.assertEqual(swap.search_token_budget, 32)

    def test_validate_swap_request_rejects_unknown_harness_name(self):
        with self.assertRaisesRegex(InvalidRequestError, "Unknown harness"):
            validate_swap_request(
                {
                    "model_dir": "/tmp/model",
                    "harness_name": "bogus",
                }
            )

    def test_validate_swap_request_rejects_unknown_search_provider(self):
        with self.assertRaisesRegex(InvalidRequestError, "Unknown search provider"):
            validate_swap_request(
                {
                    "model_dir": "/tmp/model",
                    "search_provider": "bogus",
                }
            )

    def test_validate_messages_accepts_search_role(self):
        validated = validate_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "search", "content": "facts"},
            ],
            require_user_turn=False,
            allow_empty=False,
        )

        self.assertEqual(validated[-1].role, "search")
