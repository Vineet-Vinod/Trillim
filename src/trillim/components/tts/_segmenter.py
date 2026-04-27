"""Internal text segmenter for bounded TTS synthesis chunks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from trillim.components.tts._limits import (
    HARD_TEXT_SEGMENT_CAP,
    INTER_SEGMENT_LEADIN,
    LONG_TOKEN_PLACEHOLDER,
    LONG_TOKEN_RE,
    PARAGRAPH_SPLIT_RE,
    PUNCTUATION_SPLIT_RE,
    SENTENCE_SPLIT_RE,
    TARGET_TTS_TOKENS,
)

if TYPE_CHECKING:
    from pocket_tts.conditioners.text import SentencePieceTokenizer


def load_pocket_tts_tokenizer() -> SentencePieceTokenizer:
    """Load the PocketTTS sentencepiece tokenizer without loading the model."""
    from pathlib import Path

    from pocket_tts.conditioners.text import SentencePieceTokenizer
    from pocket_tts.default_parameters import DEFAULT_VARIANT
    from pocket_tts.models import tts_model as pocket_tts_model
    from pocket_tts.utils.config import load_config

    config_path = Path(pocket_tts_model.__file__).parents[1] / f"config/{DEFAULT_VARIANT}.yaml"
    config = load_config(config_path)
    lookup = config.flow_lm.lookup_table
    return SentencePieceTokenizer(
        lookup.n_bins,
        str(lookup.tokenizer_path),
    )


def iter_text_segments(text: str, tokenizer) -> Iterator[str]:
    """Yield bounded TTS segments lazily from one validated input string."""
    sanitized_text = LONG_TOKEN_RE.sub(LONG_TOKEN_PLACEHOLDER, text)
    for paragraph in _split_with(PARAGRAPH_SPLIT_RE, sanitized_text):
        for segment in _iter_paragraph_segments(paragraph, tokenizer):
            yield _add_leadin(segment, tokenizer)


def count_tts_tokens(text: str, tokenizer) -> int:
    """Return the PocketTTS token count for one text snippet."""
    return int(tokenizer(text).tokens.shape[-1])


def _iter_paragraph_segments(paragraph: str, tokenizer) -> Iterator[str]:
    for sentence in _split_with(SENTENCE_SPLIT_RE, paragraph):
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        if _fits_segment_limits(sentence, tokenizer):
            yield sentence
            continue
        punctuation_units = _split_with(PUNCTUATION_SPLIT_RE, sentence)
        if len(punctuation_units) > 1:
            yield from _iter_grouped_segments(punctuation_units, tokenizer)
            continue
        yield from _iter_whitespace_segments(sentence, tokenizer)


def _fits_segment_limits(text: str, tokenizer) -> bool:
    return (
        len(text) <= HARD_TEXT_SEGMENT_CAP
        and count_tts_tokens(text, tokenizer) <= TARGET_TTS_TOKENS
    )


def _add_leadin(text: str, tokenizer) -> str:
    candidate = f"{INTER_SEGMENT_LEADIN}{text}"
    if _fits_segment_limits(candidate, tokenizer):
        return candidate
    return text


def _iter_grouped_segments(units: list[str], tokenizer) -> Iterator[str]:
    current = ""
    for unit in units:
        unit = " ".join(unit.split())
        if not unit:
            continue
        if not _fits_segment_limits(unit, tokenizer):
            if current:
                yield current
                current = ""
            yield from _iter_whitespace_segments(unit, tokenizer)
            continue
        candidate = unit if not current else f"{current} {unit}"
        if not current or _fits_segment_limits(candidate, tokenizer):
            current = candidate
            continue
        yield current
        current = unit
    if current:
        yield current


def _iter_whitespace_segments(text: str, tokenizer) -> Iterator[str]:
    for piece in _hard_split_unit(text, tokenizer):
        piece = piece.strip()
        if piece:
            yield piece


def _hard_split_unit(unit: str, tokenizer) -> list[str]:
    text = " ".join(unit.split())
    if not text:
        return []
    if len(text) <= HARD_TEXT_SEGMENT_CAP and count_tts_tokens(text, tokenizer) <= TARGET_TTS_TOKENS:
        return [text]
    words = text.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > HARD_TEXT_SEGMENT_CAP:
            if current:
                pieces.append(current)
                current = ""
                candidate = word
            if len(candidate) > HARD_TEXT_SEGMENT_CAP:
                pieces.extend(_slice_long_word(candidate))
                current = ""
                continue
        if current and count_tts_tokens(candidate, tokenizer) > TARGET_TTS_TOKENS:
            pieces.append(current)
            current = word
            continue
        current = candidate
    if current:
        pieces.append(current)
    return pieces


def _slice_long_word(word: str) -> list[str]:
    return [
        word[start : start + HARD_TEXT_SEGMENT_CAP]
        for start in range(0, len(word), HARD_TEXT_SEGMENT_CAP)
    ]


def _split_with(pattern, text: str) -> list[str]:
    return [piece.strip() for piece in pattern.split(text) if piece.strip()]
