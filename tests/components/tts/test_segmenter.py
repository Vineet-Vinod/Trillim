from __future__ import annotations

import unittest
from unittest.mock import patch

from trillim.components.tts._limits import HARD_TEXT_SEGMENT_CAP, TARGET_TTS_TOKENS
import trillim.components.tts._segmenter as segmenter
from trillim.components.tts._segmenter import (
    _hard_split_unit,
    _iter_grouped_segments,
    _iter_paragraph_segments,
    _iter_whitespace_segments,
    _slice_long_word,
    count_tts_tokens,
    iter_text_segments,
    load_pocket_tts_tokenizer,
)


class SegmenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_pocket_tts_tokenizer()

    def test_count_tts_tokens_uses_tokenizer_shape(self):
        self.assertGreater(count_tts_tokens("one two three", self.tokenizer), 0)

    def test_iter_text_segments_splits_sentences_and_paragraphs_with_leadin(self):
        segments = list(iter_text_segments("One. Two!\n\nThree?", self.tokenizer))

        self.assertEqual(segments, ["  One.", "  Two!", "  Three?"])

    def test_iter_text_segments_groups_punctuation_units_when_sentence_is_too_long(self):
        text = "alpha, " * (TARGET_TTS_TOKENS + 5)

        segments = list(iter_text_segments(text, self.tokenizer))

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(segment.strip() for segment in segments))
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

    def test_iter_text_segments_omits_leadin_when_it_would_exceed_limits(self):
        text = ("word " * 100).strip()
        self.assertLessEqual(len(text), HARD_TEXT_SEGMENT_CAP)
        self.assertLessEqual(count_tts_tokens(text, self.tokenizer), TARGET_TTS_TOKENS)

        segments = list(iter_text_segments(text, self.tokenizer))

        self.assertEqual(segments, [text])

    def test_internal_segmenter_empty_and_oversized_unit_branches(self):
        self.assertEqual(list(_iter_paragraph_segments("   ", self.tokenizer)), [])
        self.assertEqual(list(_iter_whitespace_segments("   ", self.tokenizer)), [])
        self.assertEqual(_hard_split_unit("   ", self.tokenizer), [])
        self.assertEqual(_hard_split_unit("hello", self.tokenizer), ["hello"])
        self.assertEqual(list(_iter_grouped_segments(["   "], self.tokenizer)), [])

        long_word = "a" * (HARD_TEXT_SEGMENT_CAP + 10)
        self.assertEqual(
            _slice_long_word(long_word),
            [long_word[:HARD_TEXT_SEGMENT_CAP], long_word[HARD_TEXT_SEGMENT_CAP:]],
        )
        self.assertEqual(_hard_split_unit(long_word, self.tokenizer), _slice_long_word(long_word))
        self.assertEqual(
            list(_iter_grouped_segments(["b" * (HARD_TEXT_SEGMENT_CAP + 1)], self.tokenizer)),
            _slice_long_word("b" * (HARD_TEXT_SEGMENT_CAP + 1)),
        )

        units = ["short.", "b" * (HARD_TEXT_SEGMENT_CAP + 1), "tail."]
        segments = list(_iter_grouped_segments(units, self.tokenizer))
        self.assertGreaterEqual(len(segments), 3)
        self.assertEqual(segments[0], "short.")
        self.assertEqual(segments[-1], "tail.")

        segments = _hard_split_unit(f"short {'b' * HARD_TEXT_SEGMENT_CAP}", self.tokenizer)
        self.assertEqual(segments[0], "short")
        self.assertEqual(segments[1], "b" * HARD_TEXT_SEGMENT_CAP)

    def test_defensive_segmenter_branches_with_filtered_helper_outputs(self):
        class EmptySentenceSplitter:
            def split(self, _text):
                return ["   "]

        with patch.object(segmenter, "SENTENCE_SPLIT_RE", EmptySentenceSplitter()):
            with patch("trillim.components.tts._segmenter._split_with", return_value=["   "]):
                self.assertEqual(list(_iter_paragraph_segments("ignored", self.tokenizer)), [])

        with patch.object(segmenter, "SENTENCE_SPLIT_RE", EmptySentenceSplitter()):
            self.assertEqual(list(_iter_paragraph_segments("ignored", self.tokenizer)), [])

        with patch("trillim.components.tts._segmenter._hard_split_unit", return_value=["   "]):
            self.assertEqual(list(_iter_whitespace_segments("ignored", self.tokenizer)), [])
