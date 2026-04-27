from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop

from trillim.components import Component
from trillim.components.stt._engine import STTEngine
from trillim.components.stt._router import build_router
from trillim.components.stt._session import STTSession, _create_stt_session
from trillim.errors import ComponentLifecycleError


class STT(Component):
    def __init__(self) -> None:
        self._engine = STTEngine()
        self._transcribe_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._owner_loop: AbstractEventLoop | None = None
        self._started = False

    def router(self):
        return build_router(self)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            self._require_owner_loop()
            if self._started:
                return
            await self._engine.start()
            self._stop_event.clear()
            self._started = True

    async def stop(self) -> None:
        self._require_owner_loop()
        async with self._lifecycle_lock:
            if not self._started and self._stop_event.is_set():
                return
            self._stop_event.set()
            self._started = False
            async with self._transcribe_lock:
                await self._engine.stop()

    def open_session(self) -> STTSession:
        self._require_owner_loop()
        if not self._started or self._stop_event.is_set():
            raise ComponentLifecycleError("STT is not running")
        return _create_stt_session(self)

    async def _transcribe(self, pcm: bytes, *, language: str | None = None) -> str:
        self._require_owner_loop()
        if not self._started or self._stop_event.is_set():
            raise ComponentLifecycleError("STT component has been stopped")
        async with self._transcribe_lock:
            if not self._started or self._stop_event.is_set():
                raise ComponentLifecycleError("STT component has been stopped")
            return await self._engine.transcribe(pcm, language=language)

    def _require_owner_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
            return
        if loop is not self._owner_loop:
            raise ComponentLifecycleError(
                "STT is bound to one event loop; create a new STT per thread/event loop"
            )
