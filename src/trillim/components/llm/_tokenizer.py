"""Tokenizer loading for the LLM component."""

from __future__ import annotations

from pathlib import Path

from trillim.errors import ModelValidationError


def load_tokenizer(model_dir: Path, *, trust_remote_code: bool):
    """Load a tokenizer from a validated model directory."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ModelValidationError("transformers is required to load tokenizers") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise ModelValidationError(
            f"Could not load tokenizer from {model_dir}"
        ) from exc
    if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
        raise ModelValidationError(
            f"Tokenizer loaded from {model_dir} is missing encode/decode methods"
        )
    return tokenizer
