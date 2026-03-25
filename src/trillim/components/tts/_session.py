"""Session handles for the TTS component."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from trillim.components.tts._limits import MAX_EMITTED_AUDIO_CHUNKS
from trillim.components.tts._validation import validate_speed
from trillim.errors import (
    OperationCancelledError,
    SessionBusyError,
    SessionClosedError,
)
from trillim.utils.filesystem import unlink_if_exists


_TTS_SESSION_OWNER_TOKEN = object()
_TTS_SESSION_CONSTRUCTION_ERROR = (
    "TTSSession cannot be constructed directly; use TTS.speak()"
)
_ALLOW_TTS_SESSION_SUBCLASS = False


def _create_tts_session(
    tts,
    *,
    text: str,
    voice: str,
    voice_kind: str,
    voice_reference: str,
    speed: float,
    cleanup_path: Path | None,
    session_worker,
) -> _TTSSession:
    return _TTSSession(
        tts,
        text=text,
        voice=voice,
        voice_kind=voice_kind,
        voice_reference=voice_reference,
        speed=speed,
        cleanup_path=cleanup_path,
        session_worker=session_worker,
        _owner_token=_TTS_SESSION_OWNER_TOKEN,
    )


class TTSSession(abc.ABC):
    """Public TTS session handle returned by ``TTS.speak()``."""

    _runtime_proxy = True

    def __init_subclass__(cls, **kwargs) -> None:
        del kwargs
        super().__init_subclass__()
        if not _ALLOW_TTS_SESSION_SUBCLASS:
            raise TypeError("TTSSession cannot be subclassed publicly")

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        if cls is TTSSession:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    @property
    @abc.abstractmethod
    def state(self) -> str:
        ...  # pragma: no cover

    @property
    @abc.abstractmethod
    def voice(self) -> str:
        ...  # pragma: no cover

    @property
    @abc.abstractmethod
    def speed(self) -> float:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def __aenter__(self) -> TTSSession:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def close(self) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def pause(self) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def resume(self) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def cancel(self) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def set_speed(self, speed: float) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def collect(self) -> bytes:
        ...  # pragma: no cover

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[bytes]:
        ...  # pragma: no cover


_ALLOW_TTS_SESSION_SUBCLASS = True


class _TTSSession(TTSSession):
    """Private concrete TTS session implementation."""

    def __new__(
        cls,
        tts=None,
        *,
        text=None,
        voice=None,
        voice_kind=None,
        voice_reference=None,
        speed=None,
        cleanup_path=None,
        session_worker=None,
        _owner_token=None,
    ):
        del tts, text, voice, voice_kind, voice_reference, speed, cleanup_path, session_worker
        if _owner_token is not _TTS_SESSION_OWNER_TOKEN:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(
        self,
        tts=None,
        *,
        text=None,
        voice=None,
        voice_kind=None,
        voice_reference=None,
        speed=None,
        cleanup_path=None,
        session_worker=None,
        _owner_token=None,
    ) -> None:
        if _owner_token is not _TTS_SESSION_OWNER_TOKEN:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        if (
            tts is None
            or text is None
            or voice is None
            or voice_kind is None
            or voice_reference is None
            or speed is None
            or session_worker is None
        ):
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        self._tts = tts
        self._text = text
        self._voice = voice
        self._voice_kind = voice_kind
        self._voice_reference = voice_reference
        self._speed = speed
        self._cleanup_path = cleanup_path
        self._session_worker = session_worker
        self._state = "running"
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=MAX_EMITTED_AUDIO_CHUNKS
        )
        self._wake_event = asyncio.Event()
        self._done_event = asyncio.Event()
        self._resume_event = asyncio.Event()
        self._consumer_mode: str | None = None
        self._consumer_active = False
        self._pause_requested = False
        self._chunk_in_flight = False
        self._error: Exception | None = None
        self._task: asyncio.Task | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def speed(self) -> float:
        return self._speed

    async def __aenter__(self) -> TTSSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self.cancel()

    async def pause(self) -> None:
        await self._tts._pause_session(self)

    async def resume(self) -> None:
        await self._tts._resume_session(self)

    async def cancel(self) -> None:
        await self._tts._cancel_session(self)

    async def set_speed(self, speed: float) -> None:
        await self._tts._set_session_speed(self, validate_speed(speed))

    async def collect(self) -> bytes:
        self._begin_consumer("collect")
        chunks: list[bytes] = []
        try:
            async for chunk in self._consume():
                chunks.append(chunk)
        finally:
            self._consumer_active = False
        return b"".join(chunks)

    def __aiter__(self) -> AsyncIterator[bytes]:
        self._begin_consumer("iterate")
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._consume():
                yield chunk
        finally:
            self._consumer_active = False

    async def _consume(self) -> AsyncIterator[bytes]:
        while True:
            while not self._audio_queue.empty():
                yield await self._audio_queue.get()
            if self._done_event.is_set():
                break
            await self._wake_event.wait()
            self._wake_event.clear()
        if self._error is not None:
            raise self._error

    async def _put_chunk(self, chunk: bytes) -> None:
        await self._audio_queue.put(chunk)
        self._wake_event.set()

    async def _finish(self, state: str, error: Exception | None = None) -> None:
        if self._done_event.is_set():
            return
        self._state = state
        self._error = error
        self._resume_event.set()
        self._done_event.set()
        self._wake_event.set()
        if self._cleanup_path is not None:
            unlink_if_exists(self._cleanup_path)
            self._cleanup_path = None

    async def _wait_for_done(self) -> None:
        await self._done_event.wait()

    def _begin_consumer(self, mode: str) -> None:
        if self._consumer_mode is None:
            self._consumer_mode = mode
        elif self._consumer_mode != mode:
            raise SessionBusyError(
                "TTSSession collect() and iteration are mutually exclusive"
            )
        if self._consumer_active:
            raise SessionBusyError("TTSSession already has an active consumer")
        self._consumer_active = True

    def _set_running(self) -> None:
        self._state = "running"
        self._resume_event.set()

    def _set_paused(self) -> None:
        self._state = "paused"
        self._resume_event.clear()

    def _mark_owner_stopped(self) -> None:
        self._state = "owner_stopped"
        self._error = SessionClosedError("TTSSession owner has stopped")
        self._resume_event.set()
        task = self._task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _cancel_error(self) -> OperationCancelledError:
        return OperationCancelledError("TTS session was cancelled")

    def __del__(self):
        task = self._task
        try:
            if task is not None and not task.done():
                task.cancel()
        except Exception:
            pass


_ALLOW_TTS_SESSION_SUBCLASS = False
