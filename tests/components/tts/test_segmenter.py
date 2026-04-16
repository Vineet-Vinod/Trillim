"""Tests for TTS text segmentation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trillim.components.tts._limits import (
    HARD_TEXT_SEGMENT_CAP,
    TARGET_TTS_TOKENS,
)
from trillim.components.tts._segmenter import (
    _add_leadin,
    _hard_split_unit,
    _iter_grouped_segments,
    _iter_paragraph_segments,
    _iter_whitespace_segments,
    _slice_long_word,
    _split_with,
    count_tts_tokens,
    iter_text_segments,
    load_pocket_tts_tokenizer,
)
from tests.components.tts.support import FakeTokenizer


class TTSSegmenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()

    def test_count_tts_tokens_uses_tokenizer_shape(self):
        self.assertEqual(count_tts_tokens("one two three", self.tokenizer), 3)

    def test_iter_text_segments_prefers_paragraph_and_sentence_boundaries(self):
        first = "one two three four five six seven eight nine ten."
        second = "eleven twelve thirteen."
        third = "alpha beta gamma delta epsilon."
        text = f"{first} {second}\n\n{third}"
        segments = list(iter_text_segments(text, self.tokenizer))
        self.assertEqual(segments, [f"  {first}", f"  {second}", f"  {third}"])

    def test_iter_text_segments_replaces_too_long_non_whitespace_tokens(self):
        word = "x" * (HARD_TEXT_SEGMENT_CAP + 10)
        segments = list(iter_text_segments(f"alpha {word} omega", self.tokenizer))
        self.assertEqual(segments, ["  alpha too-long-word-skipped omega"])
        self.assertTrue(all("x" * 51 not in segment for segment in segments))
        self.assertTrue(
            all(count_tts_tokens(segment, self.tokenizer) <= TARGET_TTS_TOKENS for segment in segments)
        )

    def test_segmenter_internal_split_helpers_cover_edge_cases(self):
        self.assertEqual(_split_with(__import__("re").compile(r"\n+"), " one \n\n two \n"), ["one", "two"])
        long_word = "x" * (HARD_TEXT_SEGMENT_CAP + 10)
        self.assertEqual(
            _slice_long_word(long_word),
            ["x" * HARD_TEXT_SEGMENT_CAP, "x" * 10],
        )
        self.assertEqual(_hard_split_unit("   ", self.tokenizer), [])
        self.assertEqual(
            _hard_split_unit(f"alpha {long_word}", self.tokenizer),
            ["alpha", "x" * HARD_TEXT_SEGMENT_CAP, "x" * 10],
        )

    def test_iter_paragraph_segments_emits_each_bounded_sentence_without_merging(self):
        first = " ".join(f"word{i}" for i in range(10)) + "."
        second = " ".join(f"more{i}" for i in range(11)) + "."
        paragraph = f"{first} {second}"
        segments = list(_iter_paragraph_segments(paragraph, self.tokenizer))
        self.assertEqual(segments, [first, second])
        self.assertEqual(list(_iter_paragraph_segments(" \n ", self.tokenizer)), [])

    def test_iter_paragraph_segments_splits_oversized_sentence_on_punctuation_greedily(self):
        first = " ".join(["a"] * 40) + ","
        second = " ".join(["b"] * 40) + ";"
        third = " ".join(["c"] * 25) + "."
        sentence = f"{first} {second} {third}"

        segments = list(_iter_paragraph_segments(sentence, self.tokenizer))

        self.assertEqual(segments, [f"{first} {second}", third])
        self.assertTrue(all(count_tts_tokens(segment, self.tokenizer) <= TARGET_TTS_TOKENS for segment in segments))

    def test_iter_paragraph_segments_falls_back_to_whitespace_when_punctuation_is_absent(self):
        sentence = " ".join(["a"] * (TARGET_TTS_TOKENS + 1))

        segments = list(_iter_paragraph_segments(sentence, self.tokenizer))

        self.assertEqual(len(segments), 2)
        self.assertEqual(count_tts_tokens(segments[0], self.tokenizer), TARGET_TTS_TOKENS)
        self.assertEqual(count_tts_tokens(segments[1], self.tokenizer), 1)

    def test_iter_grouped_segments_flushes_current_before_oversized_unit(self):
        oversized = " ".join(["b"] * (TARGET_TTS_TOKENS + 1))

        segments = list(
            _iter_grouped_segments(
                ["   ", "alpha,", oversized, "omega."],
                self.tokenizer,
            )
        )

        self.assertEqual(segments[0], "alpha,")
        self.assertEqual(count_tts_tokens(segments[1], self.tokenizer), TARGET_TTS_TOKENS)
        self.assertEqual(count_tts_tokens(segments[2], self.tokenizer), 1)
        self.assertEqual(segments[3], "omega.")

    def test_iter_grouped_segments_handles_oversized_first_unit_without_trailing_current(self):
        oversized = " ".join(["b"] * (TARGET_TTS_TOKENS + 1))

        segments = list(_iter_grouped_segments([oversized], self.tokenizer))

        self.assertEqual(len(segments), 2)
        self.assertEqual(count_tts_tokens(segments[0], self.tokenizer), TARGET_TTS_TOKENS)
        self.assertEqual(count_tts_tokens(segments[1], self.tokenizer), 1)

    def test_hard_split_unit_covers_first_word_overflow_and_reset_without_reslicing(self):
        long_word = "x" * (HARD_TEXT_SEGMENT_CAP + 10)
        capped_word = "y" * HARD_TEXT_SEGMENT_CAP

        self.assertEqual(_hard_split_unit("alpha beta", self.tokenizer), ["alpha beta"])
        self.assertEqual(
            _hard_split_unit(long_word, self.tokenizer),
            ["x" * HARD_TEXT_SEGMENT_CAP, "x" * 10],
        )
        self.assertEqual(
            _hard_split_unit(f"{capped_word} z", self.tokenizer),
            [capped_word, "z"],
        )

    def test_hard_split_unit_splits_on_token_budget_without_hard_cap_overflow(self):
        words = ["a"] * (TARGET_TTS_TOKENS + 1)
        text = " ".join(words)
        if len(text) >= HARD_TEXT_SEGMENT_CAP:
            self.skipTest("current limits trigger the hard text cap before the token budget")

        pieces = _hard_split_unit(text, self.tokenizer)

        self.assertEqual(len(pieces), 2)
        self.assertEqual(count_tts_tokens(pieces[0], self.tokenizer), TARGET_TTS_TOKENS)
        self.assertEqual(count_tts_tokens(pieces[1], self.tokenizer), 1)
        self.assertLess(len(text), HARD_TEXT_SEGMENT_CAP)

    def test_iter_whitespace_segments_skips_blank_pieces_after_hard_split(self):
        with patch(
            "trillim.components.tts._segmenter._hard_split_unit",
            return_value=["   ", "alpha"],
        ):
            segments = list(_iter_whitespace_segments("ignored", self.tokenizer))

        self.assertEqual(segments, ["alpha"])

    def test_iter_paragraph_segments_skips_blank_sentences_returned_by_splitter(self):
        with patch(
            "trillim.components.tts._segmenter._split_with",
            side_effect=[["   ", "alpha beta."], ["alpha beta."]],
        ):
            segments = list(_iter_paragraph_segments("ignored", self.tokenizer))

        self.assertEqual(segments, ["alpha beta."])

    def test_add_leadin_prefixes_only_when_candidate_stays_within_limits(self):
        self.assertEqual(_add_leadin("alpha beta.", self.tokenizer), "  alpha beta.")

        with patch(
            "trillim.components.tts._segmenter._fits_segment_limits",
            side_effect=[False],
        ):
            self.assertEqual(_add_leadin("beta gamma.", self.tokenizer), "beta gamma.")

    def test_load_pocket_tts_tokenizer_uses_lookup_configuration(self):
        captured: list[tuple[int, str]] = []

        class _SentencePieceTokenizer:
            def __init__(self, n_bins: int, tokenizer_path: str) -> None:
                captured.append((n_bins, tokenizer_path))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_file = root / "pocket_tts" / "models" / "tts_model.py"
            model_file.parent.mkdir(parents=True)
            model_file.write_text("# stub\n", encoding="utf-8")
            fake_modules = {
                "pocket_tts.conditioners.text": SimpleNamespace(
                    SentencePieceTokenizer=_SentencePieceTokenizer
                ),
                "pocket_tts.default_parameters": SimpleNamespace(DEFAULT_VARIANT="demo"),
                "pocket_tts.models": SimpleNamespace(tts_model=SimpleNamespace(__file__=str(model_file))),
                "pocket_tts.utils.config": SimpleNamespace(
                    load_config=lambda path: SimpleNamespace(
                        flow_lm=SimpleNamespace(
                            lookup_table=SimpleNamespace(
                                n_bins=7,
                                tokenizer_path=path.parent / "demo.model",
                            )
                        )
                    )
                ),
            }
            with patch.dict(sys.modules, fake_modules):
                tokenizer = load_pocket_tts_tokenizer()

        self.assertIsInstance(tokenizer, _SentencePieceTokenizer)
        self.assertEqual(captured, [(7, str(model_file.parents[1] / "config" / "demo.model"))])
