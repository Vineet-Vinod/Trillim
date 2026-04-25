from __future__ import annotations

import unittest

from trillim.components.tts._limits import HARD_TEXT_SEGMENT_CAP, TARGET_TTS_TOKENS
from trillim.components.tts._segmenter import count_tts_tokens, iter_text_segments

from tests.components.tts.support import FakeTokenizer


class SegmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()

    def test_count_tts_tokens_uses_tokenizer_shape(self):
        self.assertEqual(count_tts_tokens("one two three", self.tokenizer), 3)

    def test_iter_text_segments_splits_sentences_and_paragraphs_with_leadin(self):
        segments = list(iter_text_segments("One. Two!\n\nThree?", self.tokenizer))

        self.assertEqual(segments, ["  One.", "  Two!", "  Three?"])

    def test_iter_text_segments_groups_punctuation_units_when_sentence_is_too_long(self):
        text = "alpha, " * (TARGET_TTS_TOKENS + 5)

        segments = list(iter_text_segments(text, self.tokenizer))

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(segment.startswith("  ") for segment in segments))
        self.assertTrue(all(count_tts_tokens(segment, self.tokenizer) <= TARGET_TTS_TOKENS for segment in segments))

    def test_iter_text_segments_hard_splits_long_whitespace_text(self):
        text = " ".join(f"word{index}" for index in range(TARGET_TTS_TOKENS + 20))

        segments = list(iter_text_segments(text, self.tokenizer))

        self.assertGreater(len(segments), 1)
        self.assertEqual(" ".join(segment.strip() for segment in segments), text)
        self.assertTrue(all(count_tts_tokens(segment, self.tokenizer) <= TARGET_TTS_TOKENS for segment in segments))

    def test_iter_text_segments_replaces_very_long_tokens(self):
        segments = list(iter_text_segments("a" * 80, self.tokenizer))

        self.assertEqual(segments, ["  too-long-word-skipped"])

    def test_iter_text_segments_omits_leadin_when_it_would_exceed_char_limit(self):
        text = " ".join(["x" * 50] * 10 + ["xx"])
        self.assertEqual(len(text), HARD_TEXT_SEGMENT_CAP)

        segments = list(iter_text_segments(text, self.tokenizer))

        self.assertEqual(segments, [text])
