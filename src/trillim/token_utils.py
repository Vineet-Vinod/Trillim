# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""
Shared token decoding utility for incremental text output.

Used by both inference.py (interactive chat) and server.py (API server)
to handle the pair-decode trick needed for correct spacing with
sentencepiece tokenizers.
"""


class IncrementalDecoder:
    """Decodes token IDs one at a time, preserving inter-token spacing."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prev_token = None

    def decode(self, token_id: int) -> str:
        """Decode a single token ID to its text fragment.

        Uses pair-decoding with the previous token to preserve whitespace
        that sentencepiece tokenizers encode as part of the next token.
        """
        if self.prev_token is None:
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        else:
            pair = self.tokenizer.decode(
                [self.prev_token, token_id], skip_special_tokens=True
            )
            prev_alone = self.tokenizer.decode(
                [self.prev_token], skip_special_tokens=True
            )
            text = pair[len(prev_alone) :]
        self.prev_token = token_id
        return text

    def reset(self):
        """Reset state for a new generation sequence."""
        self.prev_token = None
