"""Session handles for the TTS component."""

from __future__ import annotations

import abc
import asyncio
import math
import random
from collections import deque
from collections.abc import AsyncIterator
from enum import Enum

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
from trillim.errors import ComponentLifecycleError, SessionBusyError


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


def _create_tts_session(
    tts,
    *,
    voice: str,
    speed: float,
) -> _TTSSession:
    return _TTSSession(
        tts,
        voice=voice,
        speed=speed,
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
        _owner_token=None,
    ):
        del tts, voice, speed
        if _owner_token is not _TTS_SESSION_OWNER_TOKEN:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(
        self,
        tts=None,
        *,
        voice=None,
        speed=None,
        _owner_token=None,
    ) -> None:
        if _owner_token is not _TTS_SESSION_OWNER_TOKEN:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        if tts is None or voice is None or speed is None:
            raise TypeError(_TTS_SESSION_CONSTRUCTION_ERROR)
        self._tts = tts
        self._voice = voice
        self._speed = speed
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
        self._voice, _voice_state = await self._tts._configure_voice(voice)

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
        if self._stopped():
            raise ComponentLifecycleError("TTS component has been stopped")
        if self._stream_active or not self._done_event.is_set():
            raise SessionBusyError("TTSSession is already synthesizing")
        self._stream_active = True
        self._clear_queue()
        self._error = None
        self._task = None
        self._done_event.clear()
        self._resume_event.set()
        self._state = _TTSSessionFSM.RUNNING
        try:
            voice = self._voice
            self._voice, voice_state = await self._tts._configure_voice(voice)
            self._task = asyncio.create_task(
                self._produce(text, voice_state=voice_state)
            )
            while True:
                if self._stopped():
                    raise ComponentLifecycleError("TTS component has been stopped")
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
            if self._task is None and not self._done_event.is_set():
                await self._finish(_TTSSessionFSM.IDLE, None, clear_queue=True)
            elif not self._done_event.is_set():
                await self._cancel_active()
            self._stream_active = False

    async def _produce(self, text: str, *, voice_state: object) -> None:
        try:
            tokenizer = await self._tts._get_tokenizer()
            segments = tuple(iter_text_segments(text, tokenizer))
            for index, segment in enumerate(segments):
                if self._stopped():
                    break
                pcm = await self._tts._synthesize_segment(segment, voice_state)
                if self._stopped():
                    break
                await self._audio_queue.put(
                    _postprocess_segment_pcm(
                        pcm,
                        text=segment,
                        speed=self._speed,
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
        self._state = state
        self._error = error
        self._resume_event.set()
        if clear_queue:
            self._clear_queue()
        self._done_event.set()

    def _clear_queue(self) -> None:
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

    def _raise_if_busy(self, field_name: str) -> None:
        if not self._done_event.is_set() or self._stream_active:
            raise SessionBusyError(f"cannot change {field_name} while synthesizing")

    def _stopped(self) -> bool:
        stop_event = getattr(self._tts, "_stop_event", None)
        return bool(stop_event is not None and stop_event.is_set())

    def __del__(self):
        task = self._task
        try:
            if task is not None and not task.done():
                task.cancel()
        except Exception:
            pass


_ALLOW_TTS_SESSION_SUBCLASS = False


def _stretch_pcm_chunk(pcm: bytes, speed: float) -> bytes:
    if speed == 1.0 or not pcm:
        return pcm
    stretcher = _StreamingPCMStretcher(speed)
    return stretcher.push(pcm) + stretcher.finish()


def _segment_pause_pcm(text: str, speed: float) -> bytes:
    base_pause_ms = _boundary_pause_ms(text)
    if base_pause_ms == 0:
        return b""
    jittered_pause_ms = base_pause_ms * random.uniform(
        BOUNDARY_PAUSE_JITTER_MIN,
        BOUNDARY_PAUSE_JITTER_MAX,
    )
    return _pcm_silence(jittered_pause_ms / speed)


def _boundary_pause_ms(text: str) -> int:
    stripped = text.rstrip()
    if not stripped:
        return 0
    if stripped[-1] in ".!?":
        return SENTENCE_BOUNDARY_PAUSE_MS
    if stripped[-1] in ",;:":
        return CLAUSE_BOUNDARY_PAUSE_MS
    return 0


def _pcm_silence(duration_ms: float) -> bytes:
    if duration_ms <= 0:
        return b""
    frame_bytes = PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
    samples = round(PCM_SAMPLE_RATE * duration_ms / 1000.0)
    return b"\x00" * (samples * frame_bytes)


def _postprocess_segment_pcm(
    pcm: bytes,
    *,
    text: str,
    speed: float,
    add_pause: bool,
) -> bytes:
    emitted = _stretch_pcm_chunk(pcm, speed)
    emitted = _apply_exponential_fade_in_pcm(emitted)
    if add_pause:
        emitted += _segment_pause_pcm(text, speed)
    return emitted


def _apply_exponential_fade_in_pcm(pcm: bytes) -> bytes:
    if not pcm:
        return pcm
    frame_bytes = PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
    if len(pcm) % frame_bytes != 0:
        return pcm
    fade_frames = min(
        len(pcm) // frame_bytes,
        round(PCM_SAMPLE_RATE * SEGMENT_FADE_IN_MS / 1000.0),
    )
    if fade_frames <= 1:
        return pcm
    scale = math.exp(SEGMENT_FADE_EXPONENT) - 1.0
    faded_pcm = bytearray(pcm)
    samples = memoryview(faded_pcm).cast("h")
    for frame_index in range(fade_frames):
        position = frame_index / (fade_frames - 1)
        gain = (math.exp(SEGMENT_FADE_EXPONENT * position) - 1.0) / scale
        sample_index = frame_index * PCM_CHANNELS
        for channel in range(PCM_CHANNELS):
            samples[sample_index + channel] = int(
                round(samples[sample_index + channel] * gain)
            )
    return bytes(faded_pcm)


class _StreamingPCMStretcher:
    def __init__(self, speed: float):
        import numpy as np

        self._np = np
        self.speed = validate_speed(speed)
        self.frame_size = 1024
        self.hop_size = self.frame_size // 4
        self._window = np.hanning(self.frame_size).astype(np.float32)
        if not np.any(self._window):
            self._window = np.ones(self.frame_size, dtype=np.float32)
        self._window_sq = self._window**2
        self._phase_advance = (
            2.0
            * np.pi
            * self.hop_size
            * np.arange(self.frame_size // 2 + 1, dtype=np.float32)
            / self.frame_size
        )
        self._input = np.zeros(0, dtype=np.float32)
        self._input_base = 0
        self._total_samples = 0
        self._next_analysis_frame = 0
        self._spectra = deque()
        self._spectra_base = 0
        self._phase = None
        self._next_time_step = 0.0
        self._processed_output_frames = 0
        self._output = np.zeros(0, dtype=np.float32)
        self._weights = np.zeros(0, dtype=np.float32)
        self._output_base = 0
        self._pending_byte = b""

    def push(self, pcm_bytes: bytes) -> bytes:
        self._append_pcm(pcm_bytes)
        self._materialize_analysis_frames(final=False)
        self._process_output_frames(final=False)
        return self._emit_ready(final=False)

    def finish(self) -> bytes:
        self._materialize_analysis_frames(final=True)
        self._process_output_frames(final=True)
        return self._emit_ready(final=True)

    def _append_pcm(self, pcm_bytes: bytes) -> None:
        np = self._np
        data = self._pending_byte + pcm_bytes
        if len(data) % 2 == 1:
            self._pending_byte = data[-1:]
            data = data[:-1]
        else:
            self._pending_byte = b""
        if not data:
            return
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        self._input = samples.copy() if self._input.size == 0 else np.concatenate((self._input, samples))
        self._total_samples += samples.size

    def _final_frame_limit(self) -> int:
        if self._total_samples == 0:
            return 0
        return max(1, self._total_samples - self.frame_size) + self.hop_size

    def _materialize_analysis_frames(self, *, final: bool) -> None:
        np = self._np
        available_end = self._input_base + self._input.size
        final_limit = self._final_frame_limit()
        while True:
            start = self._next_analysis_frame * self.hop_size
            end = start + self.frame_size
            if end <= available_end:
                local_start = start - self._input_base
                frame = self._input[local_start : local_start + self.frame_size]
            elif final and start < final_limit:
                local_start = start - self._input_base
                frame = self._input[local_start:]
                if frame.size < self.frame_size:
                    frame = np.pad(frame, (0, self.frame_size - frame.size))
            else:
                break
            spectrum = np.fft.rfft(frame * self._window).astype(np.complex64)
            self._spectra.append(spectrum)
            self._next_analysis_frame += 1
        trim_to = self._next_analysis_frame * self.hop_size
        drop = max(0, min(self._input.size, trim_to - self._input_base))
        if drop > 0:
            self._input = self._input[drop:]
            self._input_base += drop

    def _analysis_frame_count(self) -> int:
        return self._spectra_base + len(self._spectra)

    def _get_spectrum(self, frame_index: int):
        local_index = frame_index - self._spectra_base
        if local_index < 0 or local_index >= len(self._spectra):
            return None
        return self._spectra[local_index]

    def _process_output_frames(self, *, final: bool) -> None:
        np = self._np
        frame_count = self._analysis_frame_count()
        while True:
            if self._phase is None:
                first = self._get_spectrum(0)
                if first is None:
                    return
                self._phase = np.angle(first).astype(np.float32)
                magnitude = np.abs(first)
                stretched = magnitude * np.exp(1j * self._phase)
                self._add_output_frame(
                    np.fft.irfft(stretched, n=self.frame_size).real.astype(np.float32)
                )
                self._next_time_step = float(self.speed)
                continue
            step = self._next_time_step
            current_index = int(step)
            if final:
                if step >= frame_count:
                    break
            elif current_index + 1 >= frame_count:
                break
            next_index = min(current_index + 1, frame_count - 1)
            current = self._get_spectrum(current_index)
            following = self._get_spectrum(next_index)
            if current is None or following is None:
                break
            fraction = step - current_index
            current_mag = np.abs(current)
            following_mag = np.abs(following)
            magnitude = current_mag * (1.0 - fraction) + following_mag * fraction
            delta = np.angle(following) - np.angle(current) - self._phase_advance
            delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
            phase_increment = self._phase_advance + delta
            self._phase = (self._phase + phase_increment).astype(np.float32)
            stretched = magnitude * np.exp(1j * self._phase)
            self._add_output_frame(
                np.fft.irfft(stretched, n=self.frame_size).real.astype(np.float32)
            )
            self._next_time_step += self.speed

    def _add_output_frame(self, frame) -> None:
        np = self._np
        output_start = self._output_base + (self._processed_output_frames * self.hop_size)
        output_end = output_start + self.frame_size
        needed = output_end - self._output_base
        if needed > self._output.size:
            pad = needed - self._output.size
            self._output = np.pad(self._output, (0, pad))
            self._weights = np.pad(self._weights, (0, pad))
        local_start = output_start - self._output_base
        self._output[local_start : local_start + self.frame_size] += frame * self._window
        self._weights[local_start : local_start + self.frame_size] += self._window_sq
        self._processed_output_frames += 1

    def _emit_ready(self, *, final: bool) -> bytes:
        np = self._np
        if self._processed_output_frames == 0:
            return b""
        ready_frames = self._processed_output_frames if final else max(0, self._processed_output_frames - 2)
        if ready_frames <= 0:
            return b""
        local_end = ready_frames * self.hop_size
        audio = self._output[:local_end]
        weights = self._weights[:local_end]
        nonzero = weights > 1e-8
        normalized = np.zeros_like(audio)
        normalized[nonzero] = audio[nonzero] / weights[nonzero]
        clipped = np.clip(normalized, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16).tobytes()
        self._output = self._output[local_end:]
        self._weights = self._weights[local_end:]
        self._output_base += local_end
        self._drop_consumed_spectra(ready_frames)
        self._processed_output_frames -= ready_frames
        return pcm

    def _drop_consumed_spectra(self, ready_frames: int) -> None:
        keep_from = max(0, ready_frames - 1)
        while self._spectra_base < keep_from:
            if not self._spectra:
                break
            self._spectra.popleft()
            self._spectra_base += 1
