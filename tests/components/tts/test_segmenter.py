"""Tests for TTS text segmentation."""

from __future__ import annotations

import unittest

from trillim.components.tts._limits import HARD_TEXT_SEGMENT_CAP, TARGET_TTS_TOKENS
from trillim.components.tts._segmenter import count_tts_tokens, iter_text_segments
from tests.components.tts.support import FakeTokenizer


class TTSSegmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()

    def test_count_tts_tokens_uses_tokenizer_shape(self):
        self.assertEqual(count_tts_tokens("one two three", self.tokenizer), 3)

    def test_iter_text_segments_prefers_paragraph_and_sentence_boundaries(self):
        text = (
            "one two three four five six seven eight nine ten. "
            "eleven twelve thirteen.\n\n"
            "alpha beta gamma delta epsilon."
        )
        segments = list(iter_text_segments(text, self.tokenizer))
        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(all(segment.strip() == segment for segment in segments))
        self.assertIn("alpha beta gamma delta epsilon.", segments[-1])

    def test_iter_text_segments_replaces_too_long_non_whitespace_tokens(self):
        word = "x" * (HARD_TEXT_SEGMENT_CAP + 10)
        segments = list(iter_text_segments(f"alpha {word} omega", self.tokenizer))
        self.assertEqual(segments, ["alpha too-long-1word-skipped omega"])
        self.assertTrue(all("x" * 51 not in segment for segment in segments))
        self.assertTrue(
            all(count_tts_tokens(segment, self.tokenizer) <= TARGET_TTS_TOKENS for segment in segments)
        )
