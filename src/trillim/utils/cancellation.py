"""Shared cancellation primitives."""

from __future__ import annotations

import asyncio
import threading

from trillim.errors import OperationCancelledError


class CancellationToken:
    """Read-only view of a cancellation signal."""

    def __init__(self, event: threading.Event) -> None:
        """Create a token backed by a thread-safe event."""
        self._event = event

    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise if cancellation has already been requested."""
        if self.cancelled():
            raise OperationCancelledError("Operation was cancelled")

    async def wait(self) -> None:
        """Wait until cancellation is requested."""
        await asyncio.to_thread(self._event.wait)


class CancellationSource:
    """Thread-safe owner for a cancellation token."""

    def __init__(self) -> None:
        """Create a new cancellation source."""
        self._event = threading.Event()
        self.token = CancellationToken(self._event)

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

