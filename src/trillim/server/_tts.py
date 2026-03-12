# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""TTS component — text-to-speech using pocket-tts."""

import asyncio
import functools
import re
import struct
import tempfile
import threading
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ._component import Component
from ._models import (
    SpeechRequest,
    VoiceCreateResponse,
    VoiceInfo,
    VoiceListResponse,
)

_DEFAULT_VOICES_DIR = Path.home() / ".trillim" / "voices"
_DEFAULT_SPEED = 1.0
_MIN_SPEED = 0.25
_MAX_SPEED = 4.0
_PCM_STREAM_CHUNK_BYTES = 4096
_SESSION_END = object()


# ---------------------------------------------------------------------------
# WAV header utility
# ---------------------------------------------------------------------------


def wav_header(
    sample_rate: int = 24000,
    bits_per_sample: int = 16,
    channels: int = 1,
    data_size: int = 0xFFFFFFFF,
) -> bytes:
    """Build a 44-byte WAV header.

    With the default data_size=0xFFFFFFFF, produces a streaming-friendly
    header where the size fields indicate an indefinite stream.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    # RIFF chunk size = 36 + data_size (capped for streaming)
    riff_size = 36 + data_size if data_size != 0xFFFFFFFF else 0xFFFFFFFF
    header = struct.pack(
        "<4sI4s"  # RIFF header
        "4sIHHIIHH"  # fmt sub-chunk
        "4sI",  # data sub-chunk header
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def _validate_speed(speed: float) -> float:
    speed = float(speed)
    if not (_MIN_SPEED <= speed <= _MAX_SPEED):
        raise ValueError(f"speed must be between {_MIN_SPEED} and {_MAX_SPEED}")
    return speed


def _iter_pcm_chunks(
    pcm_bytes: bytes,
    chunk_size: int = _PCM_STREAM_CHUNK_BYTES,
):
    for start in range(0, len(pcm_bytes), chunk_size):
        yield pcm_bytes[start : start + chunk_size]


class _SpeedController:
    def __init__(self, speed: float):
        self._lock = threading.Lock()
        self._speed = _validate_speed(speed)

    def get(self) -> float:
        with self._lock:
            return self._speed

    def set(self, speed: float) -> float:
        validated = _validate_speed(speed)
        with self._lock:
            self._speed = validated
        return validated


class _StreamingPCMStretcher:
    def __init__(self, speed: float):
        import numpy as np

        self._np = np
        self.speed = _validate_speed(speed)
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
        self._spectra: deque = deque()
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
        if self._input.size == 0:
            self._input = samples.copy()
        else:
            self._input = np.concatenate((self._input, samples))
        self._total_samples += samples.size

    def _final_frame_limit(self) -> int:
        if self._total_samples == 0:
            return 0
        return max(1, self._total_samples - self.frame_size) + self.hop_size

    def _materialize_analysis_frames(self, final: bool) -> None:
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

    def _process_output_frames(self, final: bool) -> None:
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
            magnitude = (1.0 - fraction) * np.abs(current) + fraction * np.abs(following)
            phase_delta = np.angle(following) - np.angle(current) - self._phase_advance
            phase_delta -= 2.0 * np.pi * np.round(phase_delta / (2.0 * np.pi))
            self._phase += self._phase_advance + phase_delta
            stretched = magnitude * np.exp(1j * self._phase)
            self._add_output_frame(
                np.fft.irfft(stretched, n=self.frame_size).real.astype(np.float32)
            )
            self._next_time_step += self.speed

        trim_before = int(self._next_time_step)
        while self._spectra and self._spectra_base < trim_before:
            self._spectra.popleft()
            self._spectra_base += 1

    def _add_output_frame(self, frame) -> None:
        start = self._processed_output_frames * self.hop_size
        local_start = start - self._output_base
        if local_start < 0:
            raise RuntimeError("Streaming stretcher emitted samples out of order")
        required = local_start + self.frame_size
        if required > self._output.size:
            pad = required - self._output.size
            self._output = self._np.pad(self._output, (0, pad))
            self._weights = self._np.pad(self._weights, (0, pad))
        self._output[local_start : local_start + self.frame_size] += frame * self._window
        self._weights[local_start : local_start + self.frame_size] += self._window_sq
        self._processed_output_frames += 1

    def _emit_ready(self, final: bool) -> bytes:
        np = self._np
        if final:
            target_length = 0
            if self._total_samples > 0:
                target_length = max(1, int(round(self._total_samples / self.speed)))
            ready_count = max(0, target_length - self._output_base)
            if ready_count > self._output.size:
                pad = ready_count - self._output.size
                self._output = np.pad(self._output, (0, pad))
                self._weights = np.pad(self._weights, (0, pad))
        else:
            ready_limit = self._processed_output_frames * self.hop_size
            ready_count = max(0, min(self._output.size, ready_limit - self._output_base))

        if ready_count <= 0:
            return b""

        output = self._output[:ready_count].copy()
        weights = self._weights[:ready_count]
        nonzero = weights > 1e-6
        output[nonzero] /= weights[nonzero]
        output[~nonzero] = 0.0
        clipped = np.clip(output, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16).tobytes()
        self._output = self._output[ready_count:]
        self._weights = self._weights[ready_count:]
        self._output_base += ready_count
        return pcm


class _SessionSpeedProcessor:
    def __init__(self, controller: _SpeedController):
        self._controller = controller
        self._speed = controller.get()
        self._stretcher = None
        if self._speed != _DEFAULT_SPEED:
            self._stretcher = _StreamingPCMStretcher(self._speed)

    def push(self, raw_chunk: bytes) -> list[bytes]:
        outputs: list[bytes] = []
        target_speed = self._controller.get()
        if target_speed != self._speed:
            outputs.extend(self._flush())
            self._speed = target_speed
            self._stretcher = None
            if self._speed != _DEFAULT_SPEED:
                self._stretcher = _StreamingPCMStretcher(self._speed)

        if self._stretcher is None:
            if raw_chunk:
                outputs.append(raw_chunk)
            return outputs

        stretched = self._stretcher.push(raw_chunk)
        outputs.extend(chunk for chunk in _iter_pcm_chunks(stretched) if chunk)
        return outputs

    def finish(self) -> list[bytes]:
        return self._flush()

    def _flush(self) -> list[bytes]:
        if self._stretcher is None:
            return []
        stretched = self._stretcher.finish()
        self._stretcher = None
        return [chunk for chunk in _iter_pcm_chunks(stretched) if chunk]

# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

PREDEFINED_VOICES = frozenset(
    [
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
    ]
)


class TTSEngine:
    """Async wrapper around pocket-tts for text-to-speech.

    Custom voices are persisted as WAV files in ``voices_dir`` and
    automatically discovered on startup, so they survive server restarts.
    """

    DEFAULT_VOICE = "alba"

    def __init__(
        self,
        voices_dir: Path | None = None,
        default_voice: str = DEFAULT_VOICE,
        speed: float = _DEFAULT_SPEED,
    ):
        self._model = None
        self._voice_states: dict[str, dict] = {}
        self._custom_voice_files: dict[str, Path] = {}  # voice_id -> WAV path
        self._voices_dir = voices_dir
        self._lock = asyncio.Lock()
        self.sample_rate: int = 24000
        self.default_voice: str = default_voice
        self.speed: float = _validate_speed(speed)

    async def start(self) -> None:
        """Load the TTS model, default voice state, and discover saved voices."""
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            self._load,
        )
        self.sample_rate = self._model.sample_rate

        # Discover previously-saved custom voices (states loaded lazily)
        if self._voices_dir is not None:
            self._voices_dir.mkdir(parents=True, exist_ok=True)
            for wav in sorted(self._voices_dir.glob("*.wav")):
                voice_id = wav.stem
                if voice_id not in PREDEFINED_VOICES:
                    self._custom_voice_files[voice_id] = wav
        self._voice_states[self.default_voice] = await loop.run_in_executor(
            None,
            self._get_voice_state,
            self.default_voice,
        )

    def _load(self):
        try:
            from pocket_tts import TTSModel
        except ModuleNotFoundError as e:
            docs_path = Path(__file__).resolve().parents[1] / "docs" / "server.md"
            if not docs_path.exists():
                docs_path = Path(__file__).resolve().parents[3] / "docs" / "server.md"
            raise RuntimeError(
                "Voice support is optional and requires the 'voice' extra. "
                "Install with uv or pip using 'trillim[voice]'. "
                f"Docs: {docs_path} (section: Voice Optional Dependencies)"
            ) from e

        model = TTSModel.load_model()
        model.eval()
        return model

    async def stop(self) -> None:
        self._model = None
        self._voice_states.clear()
        self._custom_voice_files.clear()

    def _get_voice_state(self, voice: str | None) -> dict:
        voice = voice or self.default_voice
        if voice not in PREDEFINED_VOICES and voice not in self._custom_voice_files:
            raise ValueError(f"Unknown voice: {voice}")
        if voice not in self._voice_states:
            if voice in self._custom_voice_files:
                state = self._model.get_state_for_audio_prompt(
                    self._custom_voice_files[voice],
                    truncate=True,
                )
            else:
                # Predefined voice — loaded on demand
                state = self._model.get_state_for_audio_prompt(voice)
            self._voice_states[voice] = state
        return self._voice_states[voice]

    # ----- Voice management -----

    def list_voices(self) -> list[dict]:
        """Return metadata for all available voices."""
        voices = []
        for name in sorted(PREDEFINED_VOICES):
            voices.append(
                {
                    "voice_id": name,
                    "name": name,
                    "type": "predefined",
                }
            )
        for voice_id in sorted(self._custom_voice_files):
            voices.append(
                {
                    "voice_id": voice_id,
                    "name": voice_id,
                    "type": "custom",
                }
            )
        return voices

    async def register_voice(self, voice_id: str, audio_bytes: bytes) -> None:
        """Register a custom voice from uploaded audio bytes.

        The WAV file is saved to ``voices_dir/{voice_id}.wav`` so it
        persists across server restarts.  The voice state is pre-computed
        and cached in memory for immediate use.
        """
        if self._model is None:
            raise RuntimeError("TTSEngine not started")
        if voice_id in PREDEFINED_VOICES:
            raise ValueError(
                f"'{voice_id}' is a predefined voice and cannot be overwritten"
            )
        if self._voices_dir is None:
            raise RuntimeError("No voices directory configured")

        if not voice_id or voice_id in (".", "..") or "/" in voice_id or "\\" in voice_id or voice_id != Path(voice_id).name:
            raise ValueError(f"Invalid voice_id: {voice_id!r} (must be a simple filename)")
        dest = self._voices_dir / f"{voice_id}.wav"
        with tempfile.NamedTemporaryFile(
            dir=self._voices_dir,
            prefix=f"{voice_id}.",
            suffix=".wav",
            delete=False,
        ) as handle:
            handle.write(audio_bytes)
            temp_path = Path(handle.name)

        loop = asyncio.get_running_loop()
        try:
            state = await loop.run_in_executor(
                None,
                functools.partial(
                    self._model.get_state_for_audio_prompt,
                    temp_path,
                    truncate=True,
                ),
            )
            temp_path.replace(dest)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        self._custom_voice_files[voice_id] = dest
        self._voice_states[voice_id] = state

    async def delete_voice(self, voice_id: str) -> None:
        """Remove a custom voice (both the cached state and the WAV file)."""
        if voice_id in PREDEFINED_VOICES:
            raise ValueError(
                f"'{voice_id}' is a predefined voice and cannot be deleted"
            )
        path = self._custom_voice_files.pop(voice_id, None)
        if path is None:
            raise KeyError(f"Voice '{voice_id}' not found")
        path.unlink(missing_ok=True)
        self._voice_states.pop(voice_id, None)
        if self.default_voice == voice_id:
            self.default_voice = self.DEFAULT_VOICE

    # ----- Synthesis -----

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Yield PCM int16 audio chunks for the given text."""
        effective_speed = _validate_speed(self.speed if speed is None else speed)
        if effective_speed == _DEFAULT_SPEED:
            async for raw_chunk in self._synthesize_raw_stream(text, voice=voice):
                yield raw_chunk
            return

        loop = asyncio.get_running_loop()
        stretcher = _StreamingPCMStretcher(effective_speed)
        async for raw_chunk in self._synthesize_raw_stream(text, voice=voice):
            stretched = await loop.run_in_executor(
                None,
                stretcher.push,
                raw_chunk,
            )
            for chunk in _iter_pcm_chunks(stretched):
                if chunk:
                    yield chunk

        stretched = await loop.run_in_executor(None, stretcher.finish)
        for chunk in _iter_pcm_chunks(stretched):
            if chunk:
                yield chunk

    async def synthesize_full(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        """Synthesize text and return a complete WAV file as bytes."""
        chunks = []
        async for chunk in self.synthesize_stream(text, voice, speed):
            chunks.append(chunk)
        pcm_data = b"".join(chunks)
        return wav_header(self.sample_rate, data_size=len(pcm_data)) + pcm_data

    async def _synthesize_raw_stream(
        self,
        text: str,
        voice: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        if self._model is None:
            raise RuntimeError("TTSEngine not started")

        async with self._lock:
            loop = asyncio.get_running_loop()
            voice_state = await loop.run_in_executor(
                None,
                self._get_voice_state,
                voice,
            )
            gen = self._model.generate_audio_stream(
                model_state=voice_state,
                text_to_generate=text,
                copy_state=True,
            )

            while True:
                chunk_tensor = await loop.run_in_executor(
                    None,
                    functools.partial(next, gen, None),
                )
                if chunk_tensor is None:
                    break
                import numpy as np

                arr = chunk_tensor.numpy()
                pcm = np.clip(arr, -1.0, 1.0)
                pcm = (pcm * 32767).astype(np.int16)
                yield pcm.tobytes()


# ---------------------------------------------------------------------------
# SentenceChunker — buffers text and yields at sentence boundaries
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_CLAUSE_END = re.compile(r"(?<=[;:\n])\s?")

MIN_CHUNK = 10
MAX_BUFFER = 500


class SentenceChunker:
    """Buffers streaming text and yields complete sentences.

    Useful for pipelining LLM output into TTS — feed tokens as they
    arrive and synthesize each sentence without waiting for the full
    response.

    Example snippet::

        from trillim import LLM, TTS, SentenceChunker

        chunker = SentenceChunker()
        for token_text in llm_stream:
            for sentence in chunker.feed(token_text):
                await tts.synthesize_stream(sentence)
        remaining = chunker.flush()
        if remaining:
            await tts.synthesize_stream(remaining)
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Append text and return any complete sentences."""
        self._buffer += text
        results: list[str] = []

        while True:
            # Force flush if buffer exceeds max
            if len(self._buffer) > MAX_BUFFER:
                # Try to split at the last sentence boundary
                m = None
                for m in _SENTENCE_END.finditer(self._buffer):
                    pass
                if m and m.start() >= MIN_CHUNK:
                    results.append(self._buffer[: m.start() + 1].strip())
                    self._buffer = self._buffer[m.end() :]
                    continue
                # Try clause boundary
                for m in _CLAUSE_END.finditer(self._buffer):
                    pass
                if m and m.start() >= MIN_CHUNK:
                    results.append(self._buffer[: m.start() + 1].strip())
                    self._buffer = self._buffer[m.end() :]
                    continue
                # Hard flush at MAX_BUFFER
                results.append(self._buffer.strip())
                self._buffer = ""
                break

            # Look for sentence boundary
            m = _SENTENCE_END.search(self._buffer)
            if m and m.start() >= MIN_CHUNK:
                results.append(self._buffer[: m.start() + 1].strip())
                self._buffer = self._buffer[m.end() :]
                continue

            # No complete sentence yet
            break

        return results

    def flush(self) -> str | None:
        """Return any remaining buffered text."""
        text = self._buffer.strip()
        self._buffer = ""
        return text if text else None


# ---------------------------------------------------------------------------
# TTS sessions
# ---------------------------------------------------------------------------


class TTSSession:
    """Application-facing TTS session.

    Sessions are async iterable and expose flow-control methods for future
    chunk production. They do not own speaker playback.
    """

    _runtime_proxy = True

    def __init__(
        self,
        tts,
        *,
        text: str,
        voice: str | None,
        speed: float,
        timeout: float | None,
    ):
        if tts._loop is None:
            raise RuntimeError("TTS not started")
        self._tts = tts
        self._loop = tts._loop
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._chunks: asyncio.Queue[bytes | object] = asyncio.Queue()
        self._done = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._state = "queued"
        self._error: Exception | None = None
        self.text = text
        self.voice = voice
        self.speed = speed
        self._speed_controller = _SpeedController(speed)
        self.timeout = timeout
        self.sample_rate = tts.sample_rate

    @property
    def state(self) -> str:
        return self._state

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def error(self) -> Exception | None:
        return self._error

    def pause(self) -> None:
        self._schedule(self._tts._pause_session, self)

    def resume(self) -> None:
        self._schedule(self._tts._resume_session, self)

    def stop(self) -> None:
        self.cancel()

    def cancel(self) -> None:
        self._schedule(self._tts._cancel_session, self)

    def set_speed(self, speed: float) -> None:
        validated = _validate_speed(speed)
        self._schedule(self._tts._set_session_speed, self, validated)

    def _schedule(self, callback, *args) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            callback(*args)
            return
        self._loop.call_soon_threadsafe(callback, *args)

    def _set_state(self, state: str) -> None:
        if self._done.is_set():
            return
        self._state = state

    def _finish(self, state: str, error: Exception | None = None) -> None:
        if self._done.is_set():
            return
        self._state = state
        self._error = error
        self._chunks.put_nowait(_SESSION_END)
        self._done.set()

    def __aiter__(self):
        return self._stream()

    async def _stream(self):
        while True:
            item = await self._chunks.get()
            if item is _SESSION_END:
                break
            yield item
        if self._error is not None:
            raise self._error

    async def wait(self) -> None:
        await self._done.wait()
        if self._error is not None:
            raise self._error

    async def collect(self) -> bytes:
        output = bytearray()
        async for chunk in self:
            output.extend(chunk)
        return bytes(output)


# ---------------------------------------------------------------------------
# TTS component
# ---------------------------------------------------------------------------


class TTS(Component):
    """Text-to-speech component using pocket-tts."""

    def __init__(
        self,
        voices_dir: str | Path = _DEFAULT_VOICES_DIR,
        default_voice: str = TTSEngine.DEFAULT_VOICE,
        speed: float = _DEFAULT_SPEED,
    ):
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._engine = None
        self._default_voice = default_voice
        self._speed = _validate_speed(speed)
        self._loop = None
        self._active_session = None
        self._queued_sessions = deque()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._active_session = None
        self._queued_sessions.clear()
        self._engine = TTSEngine(
            voices_dir=self._voices_dir,
            default_voice=self._default_voice,
            speed=self._speed,
        )
        await self._engine.start()

    async def stop(self) -> None:
        sessions: list[TTSSession] = []
        if self._active_session is not None:
            sessions.append(self._active_session)
        sessions.extend(self._queued_sessions)
        self._cancel_all_sessions()
        if sessions:
            waiters = []
            for session in sessions:
                if session._task is not None:
                    waiters.append(session._task)
                else:
                    waiters.append(session._done.wait())
            await asyncio.gather(
                *waiters,
                return_exceptions=True,
            )
        if self._engine is not None:
            await self._engine.stop()
            self._engine = None
        self._loop = None
        self._active_session = None
        self._queued_sessions.clear()

    @property
    def engine(self):
        return self._engine

    def _require_started(self) -> TTSEngine:
        if self._engine is None:
            raise RuntimeError("TTS not started")
        return self._engine

    @staticmethod
    def _validate_input_text(text: str) -> None:
        if not text.strip():
            raise ValueError("input text is empty")

    @staticmethod
    def _validate_timeout(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        return timeout

    @property
    def default_voice(self) -> str:
        if self._engine is not None:
            return self._engine.default_voice
        return self._default_voice

    @default_voice.setter
    def default_voice(self, voice: str) -> None:
        if not voice:
            raise ValueError("default_voice must not be empty")
        if self._engine is not None:
            known = {entry["voice_id"] for entry in self._engine.list_voices()}
            if voice not in known:
                raise ValueError(f"Unknown voice: {voice}")
            self._engine.default_voice = voice
        self._default_voice = voice

    @property
    def sample_rate(self) -> int:
        return self._require_started().sample_rate

    @property
    def speed(self) -> float:
        if self._engine is not None:
            return self._engine.speed
        return self._speed

    @speed.setter
    def speed(self, speed: float) -> None:
        validated = _validate_speed(speed)
        if self._engine is not None:
            self._engine.speed = validated
        self._speed = validated

    def list_voices(self) -> list[dict]:
        return self._require_started().list_voices()

    async def register_voice(self, voice_id: str, audio_bytes: bytes) -> None:
        await self._require_started().register_voice(voice_id, audio_bytes)

    async def delete_voice(self, voice_id: str) -> None:
        engine = self._require_started()
        await engine.delete_voice(voice_id)
        if self._default_voice == voice_id:
            self._default_voice = engine.default_voice

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> AsyncGenerator[bytes, None]:
        self._validate_input_text(text)
        engine = self._require_started()
        async for chunk in engine.synthesize_stream(
            text,
            voice=voice,
            speed=speed,
        ):
            yield chunk

    async def synthesize_wav(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        self._validate_input_text(text)
        return await self._require_started().synthesize_full(
            text,
            voice=voice,
            speed=speed,
        )

    def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        timeout: float | None = None,
        interrupt: bool = False,
    ) -> TTSSession:
        self._validate_input_text(text)
        self._require_started()
        resolved_speed = self.speed if speed is None else _validate_speed(speed)
        resolved_timeout = self._validate_timeout(timeout)
        session = TTSSession(
            self,
            text=text,
            voice=voice,
            speed=resolved_speed,
            timeout=resolved_timeout,
        )
        if interrupt:
            self._cancel_all_sessions()
        if self._active_session is None:
            self._start_session(session)
        else:
            self._queued_sessions.append(session)
            session._set_state("queued")
        return session

    def _start_session(self, session: TTSSession) -> None:
        if self._loop is None:
            raise RuntimeError("TTS not started")
        self._active_session = session
        session._set_state(
            "running" if session._resume_event.is_set() else "paused"
        )
        session._task = self._loop.create_task(self._run_session(session))

    def _start_next_session(self) -> None:
        if self._active_session is not None:
            return
        while self._queued_sessions:
            session = self._queued_sessions.popleft()
            if session.done:
                continue
            self._start_session(session)
            return

    def _pause_session(self, session: TTSSession) -> None:
        if session.done:
            return
        session._resume_event.clear()
        session._set_state("paused")

    def _resume_session(self, session: TTSSession) -> None:
        if session.done:
            return
        session._resume_event.set()
        if self._active_session is session:
            session._set_state("running")
        elif session in self._queued_sessions:
            session._set_state("queued")

    def _cancel_session(self, session: TTSSession) -> None:
        if session.done:
            return
        if session is self._active_session:
            session._set_state("cancelled")
            if session._task is not None:
                session._task.cancel()
                session._task.add_done_callback(
                    lambda _task, current=session: current._finish("cancelled")
                )
            else:
                session._finish("cancelled")
            return
        try:
            self._queued_sessions.remove(session)
        except ValueError:
            pass
        session._finish("cancelled")

    def _set_session_speed(self, session: TTSSession, speed: float) -> None:
        if session.done:
            return
        session.speed = session._speed_controller.set(speed)

    def _cancel_all_sessions(self) -> None:
        if self._active_session is not None:
            active = self._active_session
            self._cancel_session(active)
            self._active_session = None
        while self._queued_sessions:
            self._queued_sessions.popleft()._finish("cancelled")

    async def _drain_session(self, session: TTSSession, iterator) -> None:
        while True:
            await session._resume_event.wait()
            if session.state == "cancelled":
                return
            try:
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            await session._resume_event.wait()
            if session.state == "cancelled":
                return
            session._chunks.put_nowait(chunk)

    async def _run_session(self, session: TTSSession) -> None:
        iterator = self._session_stream(session).__aiter__()
        try:
            if session.timeout is None:
                await self._drain_session(session, iterator)
            else:
                try:
                    async with asyncio.timeout(session.timeout):
                        await self._drain_session(session, iterator)
                except TimeoutError:
                    message = f"TTS session timed out after {session.timeout} seconds"
                    session._finish("failed", TimeoutError(message))
                    return
            if session.state != "cancelled":
                session._finish("completed")
        except asyncio.CancelledError:
            session._finish("cancelled")
        except Exception as exc:
            session._finish("failed", exc)
        finally:
            try:
                await iterator.aclose()
            except Exception:
                pass
            if self._active_session is session:
                self._active_session = None
            self._start_next_session()

    async def _session_stream(self, session: TTSSession) -> AsyncGenerator[bytes, None]:
        engine = self._require_started()
        loop = asyncio.get_running_loop()
        processor = _SessionSpeedProcessor(session._speed_controller)
        raw_iterator = engine._synthesize_raw_stream(
            session.text,
            voice=session.voice,
        ).__aiter__()

        try:
            while True:
                try:
                    raw_chunk = await raw_iterator.__anext__()
                except StopAsyncIteration:
                    break
                chunks = await loop.run_in_executor(None, processor.push, raw_chunk)
                for chunk in chunks:
                    yield chunk

            chunks = await loop.run_in_executor(None, processor.finish)
            for chunk in chunks:
                yield chunk
        finally:
            aclose = getattr(raw_iterator, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

    def router(self) -> APIRouter:
        docs_path = Path(__file__).resolve().parents[1] / "docs" / "server.md"
        if not docs_path.exists():
            docs_path = Path(__file__).resolve().parents[3] / "docs" / "server.md"
        try:
            import python_multipart  # noqa: F401
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Voice support requires the optional 'voice' extra. "
                "Install with uv or pip using 'trillim[voice]'. "
                f"Docs: {docs_path} (section: Voice Optional Dependencies)"
            ) from e

        r = APIRouter()
        tts = self

        @r.get("/v1/voices")
        async def list_voices():
            if tts._engine is None:
                raise HTTPException(status_code=503, detail="TTS engine not started")
            return VoiceListResponse(
                voices=[VoiceInfo(**v) for v in tts.list_voices()],
            )

        @r.post("/v1/voices")
        async def create_voice(
            file: UploadFile = File(...),
            voice_id: str = Form(...),
        ):
            if tts._engine is None:
                raise HTTPException(status_code=503, detail="TTS engine not started")
            audio_bytes = await file.read(8 * 1024 * 1024 + 1)
            if len(audio_bytes) > 8 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Upload exceeds 8 MB limit")
            try:
                await tts.register_voice(voice_id, audio_bytes)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return VoiceCreateResponse(voice_id=voice_id, status="created")

        @r.delete("/v1/voices/{voice_id}")
        async def delete_voice(voice_id: str):
            if tts._engine is None:
                raise HTTPException(status_code=503, detail="TTS engine not started")
            try:
                await tts.delete_voice(voice_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "deleted", "voice_id": voice_id}

        @r.post("/v1/audio/speech")
        async def speech(req: SpeechRequest):
            if tts._engine is None:
                raise HTTPException(status_code=503, detail="TTS engine not started")
            if not req.input.strip():
                raise HTTPException(status_code=400, detail="input text is empty")
            try:
                speed = _validate_speed(req.speed)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            if req.voice is not None:
                known_voices = {
                    entry["voice_id"] for entry in tts.list_voices()
                }
                if req.voice not in known_voices:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown voice: {req.voice}",
                    )

            if req.response_format == "pcm":
                return StreamingResponse(
                    tts.synthesize_stream(
                        req.input,
                        voice=req.voice,
                        speed=speed,
                    ),
                    media_type="audio/pcm",
                )

            async def _wav_stream():
                yield wav_header(tts.sample_rate)
                async for chunk in tts.synthesize_stream(
                    req.input,
                    voice=req.voice,
                    speed=speed,
                ):
                    yield chunk

            return StreamingResponse(_wav_stream(), media_type="audio/wav")

        return r
