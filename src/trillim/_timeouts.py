# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Internal helpers for consistent SDK timeouts."""

from __future__ import annotations

import asyncio


async def run_with_timeout(awaitable, timeout: float | None, operation: str):
    """Await *awaitable* with an optional wall-clock timeout."""
    if timeout is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"{operation} timed out after {timeout} seconds."
        ) from exc
