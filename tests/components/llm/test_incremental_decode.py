"""Tests for incremental LLM decoding."""

import unittest

from trillim.components.llm._incremental_decode import IncrementalDecoder


class _PairTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        mapping = {
            (65,): "A",
            (66,): "B",
            (65, 66): "A B",
        }
        return mapping[tuple(token_ids)]


class IncrementalDecoderTests(unittest.TestCase):
    def test_pair_decoding_preserves_spacing(self):
        decoder = IncrementalDecoder(_PairTokenizer())

        self.assertEqual(decoder.decode(65), "A")
        self.assertEqual(decoder.decode(66), " B")
        decoder.reset()
        self.assertEqual(decoder.decode(65), "A")
