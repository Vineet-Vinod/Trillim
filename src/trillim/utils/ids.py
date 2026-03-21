"""Shared stable identifier helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_id(
    prefix: str,
    value: str | bytes | Path,
    *,
    digest_size: int = 12,
) -> str:
    """Return a deterministic identifier with the given prefix."""
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("prefix must be non-empty and contain only letters, digits, or '_'")
    if digest_size < 4:
        raise ValueError("digest_size must be >= 4")
    if isinstance(value, Path):
        payload = str(value).encode("utf-8")
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = value
    digest = hashlib.blake2b(payload, digest_size=digest_size).hexdigest()
    return f"{prefix}_{digest}"

