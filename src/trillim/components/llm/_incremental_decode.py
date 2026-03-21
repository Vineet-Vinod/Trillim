"""Incremental token decoding helpers."""

from __future__ import annotations


class IncrementalDecoder:
    """Decode token IDs one at a time while preserving spacing."""

    def __init__(self, tokenizer) -> None:
        """Create a decoder for one generation stream."""
        self._tokenizer = tokenizer
        self._previous_token: int | None = None

    def decode(self, token_id: int) -> str:
        """Decode a token ID using pair decoding when needed."""
        if self._previous_token is None:
            text = self._tokenizer.decode([token_id], skip_special_tokens=True)
        else:
            pair = self._tokenizer.decode(
                [self._previous_token, token_id],
                skip_special_tokens=True,
            )
            prefix = self._tokenizer.decode(
                [self._previous_token],
                skip_special_tokens=True,
            )
            text = pair[len(prefix) :]
        self._previous_token = token_id
        return text

    def reset(self) -> None:
        """Reset decoder state for a new generation."""
        self._previous_token = None
