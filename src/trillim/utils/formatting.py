"""Formatting helpers shared across Trillim surfaces."""

from __future__ import annotations


def human_size(size_bytes: int) -> str:
    """Format one byte count using base-1024 human units."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            if unit == "B":
                return f"{size:.0f} {unit}"
            formatted = f"{size:.1f}".rstrip("0").rstrip(".")
            return f"{formatted} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
