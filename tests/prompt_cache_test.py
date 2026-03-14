# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for backend prompt-cache planning and commit helpers."""

import unittest

from trillim._prompt_cache import PromptCacheManager, PromptSnapshot


class PromptCacheManagerTests(unittest.TestCase):
    def test_restore_and_clear_reset_state(self):
        cache = PromptCacheManager()
        cache.restore(
            PromptSnapshot.create([1, 2], "ab"),
            last_cache_hit=2,
        )

        self.assertEqual(cache.token_ids, (1, 2))
        self.assertEqual(cache.prompt_str, "ab")
        self.assertEqual(cache.last_cache_hit, 2)

        cache.restore(None)
        self.assertEqual(cache.token_ids, ())
        self.assertIsNone(cache.prompt_str)
        self.assertEqual(cache.last_cache_hit, 0)

    def test_plan_reuses_string_prefix_with_exact_cached_prompt(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 2], "ab"))

        plan = cache.plan(
            PromptSnapshot.create([1, 2, 99, 100], "abcd"),
            encode_suffix=lambda suffix: [ord(ch) for ch in suffix],
        )

        self.assertEqual(plan.delta_tokens, (99, 100))
        self.assertEqual(plan.reset_flag, 0)
        self.assertEqual(plan.cache_hit, 2)

    def test_plan_resets_chat_request_without_exact_cached_prompt_string(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 2]))

        plan = cache.plan(
            PromptSnapshot.create([1, 2, 3], "chat"),
            encode_suffix=lambda suffix: [ord(ch) for ch in suffix],
        )

        self.assertEqual(plan.delta_tokens, (1, 2, 3))
        self.assertEqual(plan.reset_flag, 1)
        self.assertEqual(plan.cache_hit, 0)

    def test_plan_reuses_token_prefix_for_raw_requests(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 2]))

        plan = cache.plan(
            PromptSnapshot.create([1, 2, 3]),
            encode_suffix=lambda suffix: [],
        )

        self.assertEqual(plan.delta_tokens, (3,))
        self.assertEqual(plan.reset_flag, 0)
        self.assertEqual(plan.cache_hit, 2)

    def test_plan_resets_token_request_on_partial_match(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 9]))

        plan = cache.plan(
            PromptSnapshot.create([1, 2]),
            encode_suffix=lambda suffix: [],
        )

        self.assertEqual(plan.delta_tokens, (1, 2))
        self.assertEqual(plan.reset_flag, 1)
        self.assertEqual(plan.cache_hit, 0)

    def test_commit_generation_retains_request_prompt_when_kv_matches_request_length(self):
        cache = PromptCacheManager()
        plan = cache.plan(
            PromptSnapshot.create([1, 2], "ab"),
            encode_suffix=lambda suffix: [ord(ch) for ch in suffix],
        )

        cache.commit_generation(plan, generated_token_ids=[0], kv_position=2)

        self.assertEqual(cache.token_ids, (1, 2))
        self.assertEqual(cache.prompt_str, "ab")
        self.assertEqual(cache.last_cache_hit, 0)

    def test_commit_generation_requires_finalize_for_generated_chat_cache(self):
        cache = PromptCacheManager()
        plan = cache.plan(
            PromptSnapshot.create([1, 2], "ab"),
            encode_suffix=lambda suffix: [ord(ch) for ch in suffix],
        )

        cache.commit_generation(plan, generated_token_ids=[65], kv_position=3)

        self.assertEqual(cache.token_ids, (1, 2, 65))
        self.assertIsNone(cache.prompt_str)
        self.assertTrue(
            cache.finalize_prompt(PromptSnapshot.create([1, 2, 65], "abA"))
        )
        self.assertEqual(cache.prompt_str, "abA")

    def test_finalize_prompt_clears_prompt_string_on_token_mismatch(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 2], "ab"))

        self.assertFalse(
            cache.finalize_prompt(PromptSnapshot.create([1, 2, 3], "abc"))
        )
        self.assertIsNone(cache.prompt_str)


if __name__ == "__main__":
    unittest.main()
