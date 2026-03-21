"""Filesystem helpers shared across components."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def canonicalize_path(path: str | Path, *, strict: bool = False) -> Path:
    """Expand user markers and resolve the path."""
    return Path(path).expanduser().resolve(strict=strict)


def ensure_within_root(
    path: str | Path,
    root: str | Path,
    *,
    strict: bool = False,
) -> Path:
    """Resolve *path* and ensure it stays within *root*."""
    resolved_root = canonicalize_path(root, strict=False)
    resolved_path = canonicalize_path(path, strict=strict)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{resolved_path}' is outside allowed root '{resolved_root}'"
        ) from exc
    return resolved_path


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    mode: int = 0o600,
) -> Path:
    """Atomically write bytes to *path* using a same-directory temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(temp_path, mode)
        os.replace(temp_path, target)
        return target
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def unlink_if_exists(path: str | Path) -> None:
    """Remove *path* if it exists."""
    Path(path).unlink(missing_ok=True)

