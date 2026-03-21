"""Admission control for the STT component."""

from __future__ import annotations

import asyncio

from trillim.components.stt._limits import MAX_ACTIVE_TRANSCRIPTIONS
from trillim.errors import AdmissionRejectedError


class _AdmissionLease:
    def __init__(self, controller: TranscriptionAdmission) -> None:
        self._controller = controller
        self._released = False

    async def __aenter__(self) -> _AdmissionLease:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._controller._release()


class TranscriptionAdmission:
    """Bound STT work to one active transcription and support draining."""

    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._active = 0
        self._accepting = True
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active_count(self) -> int:
        """Return the number of active transcriptions."""
        return self._active

    @property
    def accepting(self) -> bool:
        """Return whether STT is admitting new transcriptions."""
        return self._accepting

    async def acquire(self) -> _AdmissionLease:
        """Acquire the only STT slot or fail fast."""
        async with self._state_lock:
            if not self._accepting:
                raise AdmissionRejectedError("STT is draining and not accepting new requests")
            if self._active >= MAX_ACTIVE_TRANSCRIPTIONS:
                raise AdmissionRejectedError("STT is busy")
            self._active += 1
            self._idle.clear()
            return _AdmissionLease(self)

    async def start_draining(self) -> None:
        """Reject new work while letting the in-flight request finish."""
        async with self._state_lock:
            self._accepting = False
            if self._active == 0:
                self._idle.set()

    async def finish_starting(self) -> None:
        """Resume admissions after start or restart."""
        async with self._state_lock:
            self._accepting = True
            if self._active == 0:
                self._idle.set()

    async def wait_for_idle(self) -> None:
        """Wait for the in-flight request, if any, to finish cleanup."""
        await self._idle.wait()

    async def _release(self) -> None:
        async with self._state_lock:
            if self._active > 0:
                self._active -= 1
            if self._active == 0:
                self._idle.set()
