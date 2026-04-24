"""Session handles for the TTS component."""

from __future__ import annotations

import abc
import asyncio
import math
import random
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum
from pathlib import Path

from trillim.components.tts._limits import (
    BOUNDARY_PAUSE_JITTER_MAX,
    BOUNDARY_PAUSE_JITTER_MIN,
    CLAUSE_BOUNDARY_PAUSE_MS,
    MAX_EMITTED_AUDIO_CHUNKS,
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    SEGMENT_FADE_EXPONENT,
    SEGMENT_FADE_IN_MS,
    SENTENCE_BOUNDARY_PAUSE_MS,
)
from trillim.components.tts._segmenter import iter_text_segments
from trillim.components.tts._validation import validate_speed, validate_text
from trillim.errors import SessionBusyError
from trillim.utils.filesystem import unlink_if_exists


_TTS_SESSION_OWNER_TOKEN = object()
_TTS_SESSION_CONSTRUCTION_ERROR = (
    "TTSSession cannot be constructed directly; use TTS.open_session()"
)
_ALLOW_TTS_SESSION_SUBCLASS = False

class _TTSSessionFSM(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"


SynthesizeSegment = Callable[[str, object, float], Awaitable[bytes]]
ResolveVoice = Callable[[str], Awaitable[tuple[object, Path | None]]]
TokenizerLoader = Callable[[], Awaitable[object]]


def _create_tts_session(
    tts,
    *,
    voice: str,
    speed: float,
    resolve_voice: ResolveVoice,
    tokenizer_loader: TokenizerLoader,
    synthesize_segment: SynthesizeSegment,
) -> _TTSSession:
    return _TTSSession(
        tts,
        voice=voice,
        speed=speed,
        resolve_voice=resolve_voice,
        tokenizer_loader=tokenizer_loader,
        synthesize_segment=synthesize_segment,
        _owner_token=_TTS_SESSION_OWNER_TOKEN,
    )


class TTSSession(abc.ABC):
    """Public reusable TTS session handle returned by the TTS component."""

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
    async def set_voice(self, voice: str) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def set_speed(self, speed: float) -> None:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def collect(self, text: str) -> bytes:
        ...  # pragma: no cover

    @abc.abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        ...  # pragma: no cover


_ALLOW_TTS_SESSION_SUBCLASS = True


class _TTSSession(TTSSession):
    """Private concrete reusable TTS session implementation."""

    def __new__(
        cls,
        tts=None,
        *,
        voice=None,
        speed=None,
        resolve_voice=None,
        tokenizer_loader=None,
        synthesize_segment=None,
        _owner_token=None,
    ):
        del tts, voice, speed, resolve_voice, tokenizer_loader, synthesize_segment
        if _owner_token is not _TTS_SESSION_OWNER_TOKEN:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(
        self,
        tts=None,
        *,
        voice=None,
        speed=None,
        resolve_voice=None,
        tokenizer_loader=None,
        synthesize_segment=None,
        _owner_token=None,
    ) -> None:
        if _owner_token is not _TTS_SESSION_OWNER_TOKEN:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        if (
            tts is None
            or voice is None
            or speed is None
            or resolve_voice is None
            or tokenizer_loader is None
            or synthesize_segment is None
        ):
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        self._tts = tts
        self._voice = voice
        self._speed = speed
        self._resolve_voice = resolve_voice
        self._tokenizer_loader = tokenizer_loader
        self._synthesize_segment = synthesize_segment
        self._state = _TTSSessionFSM.IDLE
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=MAX_EMITTED_AUDIO_CHUNKS
        )
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._done_event = asyncio.Event()
        self._done_event.set()
        self._task: asyncio.Task | None = None
        self._error: Exception | None = None
        self._cleanup_path: Path | None = None
        self._stream_active = False

    @property
    def state(self) -> str:
        return self._state.value

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
        await self._cancel_active()

    async def pause(self) -> None:
        if self._state is _TTSSessionFSM.RUNNING:
            self._state = _TTSSessionFSM.PAUSED
            self._resume_event.clear()

    async def resume(self) -> None:
        if self._state is _TTSSessionFSM.PAUSED:
            self._state = _TTSSessionFSM.RUNNING
            self._resume_event.set()

    async def set_voice(self, voice: str) -> None:
        self._raise_if_busy("voice")
        self._voice = await self._tts._normalize_voice_name(voice)

    async def set_speed(self, speed: float) -> None:
        self._speed = validate_speed(speed)

    async def collect(self, text: str) -> bytes:
        chunks: list[bytes] = []
        async for chunk in self.synthesize(text):
            chunks.append(chunk)
        return b"".join(chunks)

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        return self._synthesize(validate_text(text))

    async def _synthesize(self, text: str) -> AsyncIterator[bytes]:
        if self._stream_active or not self._done_event.is_set():
            raise SessionBusyError("TTSSession is already synthesizing")
        self._stream_active = True
        self._clear_queue()
        self._error = None
        self._done_event.clear()
        self._resume_event.set()
        self._state = _TTSSessionFSM.RUNNING
        voice = self._voice
        speed = self._speed
        voice_state, cleanup_path = await self._resolve_voice(voice)
        self._cleanup_path = cleanup_path
        self._task = asyncio.create_task(
            self._produce(text, voice_state=voice_state, speed=speed)
        )
        try:
            while True:
                if self._state is _TTSSessionFSM.PAUSED:
                    await self._resume_event.wait()
                    continue
                if not self._audio_queue.empty():
                    yield await self._audio_queue.get()
                    continue
                if self._done_event.is_set():
                    break
                # If producer is producing, but queue is empty now, just sleep for
                # a bit till the next chunk comes in, rather than infinitely spinning
                await asyncio.sleep(0.5)
            if self._error is not None:
                raise self._error
        finally:
            self._stream_active = False

    async def _produce(self, text: str, *, voice_state: object, speed: float) -> None:
        try:
            tokenizer = await self._tokenizer_loader()
            segments = tuple(iter_text_segments(text, tokenizer))
            for index, segment in enumerate(segments):
                pcm = await self._synthesize_segment(segment, voice_state, speed)
                await self._audio_queue.put(
                    _postprocess_segment_pcm(
                        pcm,
                        text=segment,
                        speed=speed,
                        add_pause=index < len(segments) - 1,
                    )
                )
            await self._finish(_TTSSessionFSM.DONE, None)
        except asyncio.CancelledError:
            await self._finish(
                _TTSSessionFSM.DONE,
                None,
                clear_queue=True,
            )
            raise
        except Exception as exc:
            await self._finish(_TTSSessionFSM.IDLE, exc)

    async def _cancel_active(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._finish(
            _TTSSessionFSM.DONE,
            None,
            clear_queue=True,
        )

    async def _finish(
        self,
        state: _TTSSessionFSM,
        error: Exception | None,
        *,
        clear_queue: bool = False,
    ) -> None:
        if clear_queue:
            self._clear_queue()
        self._state = state
        self._error = error
        self._resume_event.set()
        self._done_event.set()
        if self._cleanup_path is not None:
            unlink_if_exists(self._cleanup_path)
            self._cleanup_path = None

    def _clear_queue(self) -> None:
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

    def _raise_if_busy(self, field_name: str) -> None:
        if not self._done_event.is_set() or self._stream_active:
            raise SessionBusyError(f"cannot change {field_name} while synthesizing")

    def __del__(self):
        task = self._task
        try:
            if task is not None and not task.done():
                task.cancel()
        except Exception:
            pass


_ALLOW_TTS_SESSION_SUBCLASS = False
