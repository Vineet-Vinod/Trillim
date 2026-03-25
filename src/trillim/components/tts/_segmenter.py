"""Internal text segmenter for bounded TTS synthesis chunks."""

from __future__ import annotations

import re
from collections.abc import Iterator

from trillim.components.tts._limits import (
    HARD_TEXT_SEGMENT_CAP,
    MIN_USEFUL_TTS_TOKENS,
    TARGET_TTS_TOKENS,
)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[;:])\s+")
_LINE_SPLIT_RE = re.compile(r"\n+")
_LONG_TOKEN_RE = re.compile(r"\S{51,}")
_LONG_TOKEN_PLACEHOLDER = "too-long-word-skipped"


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
    sanitized_text = _LONG_TOKEN_RE.sub(_LONG_TOKEN_PLACEHOLDER, text)
    for paragraph in _split_with(_PARAGRAPH_SPLIT_RE, sanitized_text):
        yield from _iter_paragraph_segments(paragraph, tokenizer)


def count_tts_tokens(text: str, tokenizer) -> int:
    """Return the PocketTTS token count for one text snippet."""
    return int(tokenizer(text).tokens.shape[-1])


def _iter_paragraph_segments(paragraph: str, tokenizer) -> Iterator[str]:
    units: list[str] = []
    for sentence in _split_with(_SENTENCE_SPLIT_RE, paragraph):
        for clause in _split_with(_CLAUSE_SPLIT_RE, sentence):
            units.extend(_split_with(_LINE_SPLIT_RE, clause))
    current = ""
    current_tokens = 0
    for unit in units:
        pieces = _hard_split_unit(unit, tokenizer)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            piece_tokens = count_tts_tokens(piece, tokenizer)
            if not current:
                current = piece
                current_tokens = piece_tokens
                continue
            combined = f"{current} {piece}"
            if len(combined) > HARD_TEXT_SEGMENT_CAP:
                yield current
                current = piece
                current_tokens = piece_tokens
                continue
            combined_tokens = count_tts_tokens(combined, tokenizer)
            if current_tokens >= MIN_USEFUL_TTS_TOKENS and combined_tokens > TARGET_TTS_TOKENS:
                yield current
                current = piece
                current_tokens = piece_tokens
                continue
            current = combined
            current_tokens = combined_tokens
    if current:
        yield current


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


def _split_with(pattern: re.Pattern[str], text: str) -> list[str]:
    return [piece.strip() for piece in pattern.split(text) if piece.strip()]
