"""Tests for LLM request validation."""

import unittest

from trillim.components.llm._limits import MAX_MESSAGE_CHARS, MAX_MESSAGES
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
                "num_threads": 4,
                "lora_dir": "/tmp/adapter",
                "lora_quant": "q4_0",
                "unembed_quant": "q8_0",
                "harness_name": "search",
                "search_provider": "BRAVE_SEARCH",
                "search_token_budget": 32,
            }
        )

        self.assertEqual(sampling.max_tokens, 32)
        self.assertEqual(swap.model_dir, "/tmp/model")
        self.assertEqual(swap.num_threads, 4)
        self.assertEqual(swap.lora_dir, "/tmp/adapter")
        self.assertEqual(swap.lora_quant, "q4_0")
        self.assertEqual(swap.unembed_quant, "q8_0")
        self.assertEqual(swap.harness_name, "search")
        self.assertEqual(swap.search_provider, "BRAVE_SEARCH")
        self.assertEqual(swap.search_token_budget, 32)

    def test_validate_sampling_options_accepts_runtime_lookback_range(self):
        sampling = validate_sampling_options(rep_penalty_lookback=0)
        large_sampling = validate_sampling_options(rep_penalty_lookback=512)

        self.assertEqual(sampling.rep_penalty_lookback, 0)
        self.assertEqual(large_sampling.rep_penalty_lookback, 512)

    def test_validate_sampling_options_accepts_zero_max_tokens_as_unlimited(self):
        sampling = validate_sampling_options(max_tokens=0)
        request = validate_chat_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 0,
            },
            active_model_name=None,
        )

        self.assertEqual(sampling.max_tokens, 0)
        self.assertEqual(request.max_tokens, 0)

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

    def test_validate_swap_request_rejects_blank_init_strings(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be blank"):
            validate_swap_request(
                {
                    "model_dir": "/tmp/model",
                    "lora_quant": "   ",
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

    def test_validate_chat_request_rejects_empty_and_oversized_message_content(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_chat_request(
                {
                    "messages": [{"role": "user", "content": ""}],
                },
                active_model_name=None,
            )

        with self.assertRaisesRegex(InvalidRequestError, "character limit"):
            validate_chat_request(
                {
                    "messages": [{"role": "user", "content": "x" * (MAX_MESSAGE_CHARS + 1)}],
                },
                active_model_name=None,
            )

    def test_validate_chat_request_rejects_empty_and_oversized_message_lists(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_chat_request(
                {"messages": []},
                active_model_name=None,
            )

        with self.assertRaisesRegex(InvalidRequestError, "messages exceed the limit"):
            validate_chat_request(
                {
                    "messages": [
                        {"role": "user", "content": f"message-{index}"}
                        for index in range(MAX_MESSAGES + 1)
                    ],
                },
                active_model_name=None,
            )

    def test_validate_messages_rejects_empty_and_oversized_sdk_sequences(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_messages(
                [],
                require_user_turn=False,
                allow_empty=False,
            )

        with self.assertRaisesRegex(InvalidRequestError, "messages exceed the limit"):
            validate_messages(
                [
                    {"role": "user", "content": f"message-{index}"}
                    for index in range(MAX_MESSAGES + 1)
                ],
                require_user_turn=False,
                allow_empty=True,
            )
