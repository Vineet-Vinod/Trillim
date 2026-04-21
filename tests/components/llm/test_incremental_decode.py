"""Tests for incremental LLM decoding."""

import unittest

from trillim.components.llm._incremental_decode import IncrementalDecoder


class _PairTokenizer:
    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        mapping = {
            (65,): "A",
            (66,): "B",
            (65, 66): "A B",
        }
        return mapping[tuple(token_ids)]


class _SplitUnicodeTokenizer:
    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        mapping = {
            (1,): "###",
            (1, 2): "### �",
            (1, 2, 3): "### 🔍",
            (1, 2, 3, 4): "### 🔍 heading",
        }
        return mapping[tuple(token_ids)]


class _MultiReplacementTokenizer:
    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        mapping = {
            (1,): "prefix",
            (1, 2): "prefix ��",
            (1, 2, 3): "prefix 🫠",
            (1, 2, 3, 4): "prefix 🫠 done",
        }
        return mapping[tuple(token_ids)]


class _JoinerEmojiTokenizer:
    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        mapping = {
            (1,): "status",
            (1, 2): "status 👩",
            (1, 2, 3): "status 👩‍",
            (1, 2, 3, 4): "status 👩‍💻",
            (1, 2, 3, 4, 5): "status 👩‍💻 ready",
        }
        return mapping[tuple(token_ids)]


class _CombiningMarkTokenizer:
    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        mapping = {
            (1,): "cafe",
            (1, 2): "café",
            (1, 2, 3): "café menu",
        }
        return mapping[tuple(token_ids)]


class _LongStreamTokenizer:
    def __init__(self) -> None:
        self.decode_calls = 0
        self.decoded_token_count = 0

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        self.decode_calls += 1
        self.decoded_token_count += len(token_ids)
        return "".join(f" token-{token_id}" for token_id in token_ids)


class IncrementalDecoderTests(unittest.TestCase):
    def test_pair_decoding_preserves_spacing(self):
        decoder = IncrementalDecoder(_PairTokenizer())

        self.assertEqual(decoder.decode(65), "A")
        self.assertEqual(decoder.decode(66), " B")
        decoder.reset()
        self.assertEqual(decoder.decode(65), "A")

    def test_decode_waits_for_split_unicode_tokens(self):
        decoder = IncrementalDecoder(_SplitUnicodeTokenizer())

        self.assertEqual(decoder.decode(1), "###")
        self.assertEqual(decoder.decode(2), " ")
        self.assertEqual(decoder.decode(3), "🔍")
        self.assertEqual(decoder.decode(4), " heading")

    def test_decode_strips_full_trailing_replacement_suffix(self):
        decoder = IncrementalDecoder(_MultiReplacementTokenizer())

        self.assertEqual(decoder.decode(1), "prefix")
        self.assertEqual(decoder.decode(2), " ")
        self.assertEqual(decoder.decode(3), "🫠")
        self.assertEqual(decoder.decode(4), " done")

    def test_decode_preserves_zero_width_joiner_emoji_sequences(self):
        decoder = IncrementalDecoder(_JoinerEmojiTokenizer())

        chunks = [decoder.decode(token_id) for token_id in (1, 2, 3, 4, 5)]

        self.assertEqual("".join(chunks), "status 👩‍💻 ready")
        self.assertNotIn("\ufffd", chunks)

    def test_decode_preserves_combining_marks(self):
        decoder = IncrementalDecoder(_CombiningMarkTokenizer())

        chunks = [decoder.decode(token_id) for token_id in (1, 2, 3)]

        self.assertEqual("".join(chunks), "café menu")

    def test_decode_long_stream_uses_bounded_decode_work(self):
        tokenizer = _LongStreamTokenizer()
        decoder = IncrementalDecoder(tokenizer)

        chunks = [decoder.decode(token_id) for token_id in range(1, 2_001)]

        self.assertEqual("".join(chunks).split()[:3], ["token-1", "token-2", "token-3"])
        self.assertLess(tokenizer.decoded_token_count, 64 * tokenizer.decode_calls)
