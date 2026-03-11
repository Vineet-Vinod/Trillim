# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""TTS component — text-to-speech using pocket-tts."""

import asyncio
import functools
import re
import struct
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


def _select_frame_size(sample_count: int) -> int:
    frame_size = 1024
    while frame_size > sample_count and frame_size > 64:
        frame_size //= 2
    return max(32, min(frame_size, sample_count))


def _stretch_samples(samples, speed: float):
    import numpy as np

    if samples.size < 32 or speed == _DEFAULT_SPEED:
        return samples.copy()

    frame_size = _select_frame_size(int(samples.size))
    if frame_size <= 1:
        return samples.copy()

    hop_size = max(1, frame_size // 4)
    window = np.hanning(frame_size).astype(np.float32)
    if not np.any(window):
        window = np.ones(frame_size, dtype=np.float32)

    expected_length = max(1, int(round(samples.size / speed)))
    frame_count = max(1, int(np.ceil(max(1, samples.size - frame_size) / hop_size)) + 1)
    if frame_count < 2:
        if samples.size > expected_length:
            return samples[:expected_length].copy()
        if samples.size < expected_length:
            return np.pad(samples, (0, expected_length - samples.size))
        return samples.copy()

    stft = np.empty((frame_size // 2 + 1, frame_count), dtype=np.complex64)
    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = samples[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        stft[:, frame_index] = np.fft.rfft(frame * window)

    time_steps = np.arange(0, frame_count, speed, dtype=np.float32)
    stretched_spec = np.empty((stft.shape[0], time_steps.size), dtype=np.complex64)
    phase_advance = (
        2.0
        * np.pi
        * hop_size
        * np.arange(stft.shape[0], dtype=np.float32)
        / frame_size
    )
    phase = np.angle(stft[:, 0])
    first_magnitude = np.abs(stft[:, 0])
    stretched_spec[:, 0] = first_magnitude * np.exp(1j * phase)

    for out_index, step in enumerate(time_steps[1:], start=1):
        frame_index = min(int(step), frame_count - 1)
        next_index = min(frame_index + 1, frame_count - 1)
        fraction = step - frame_index

        current = stft[:, frame_index]
        following = stft[:, next_index]
        magnitude = (1.0 - fraction) * np.abs(current) + fraction * np.abs(following)

        phase_delta = np.angle(following) - np.angle(current) - phase_advance
        phase_delta -= 2.0 * np.pi * np.round(phase_delta / (2.0 * np.pi))
        phase += phase_advance + phase_delta
        stretched_spec[:, out_index] = magnitude * np.exp(1j * phase)

    out_len = frame_size + hop_size * max(0, stretched_spec.shape[1] - 1)
    stretched = np.zeros(out_len, dtype=np.float32)
    weights = np.zeros(out_len, dtype=np.float32)
    for frame_index in range(stretched_spec.shape[1]):
        start = frame_index * hop_size
        frame = np.fft.irfft(stretched_spec[:, frame_index], n=frame_size).real
        stretched[start : start + frame_size] += frame * window
        weights[start : start + frame_size] += window**2

    nonzero = weights > 1e-6
    stretched[nonzero] /= weights[nonzero]

    if stretched.size > expected_length:
        return stretched[:expected_length]
    if stretched.size < expected_length:
        return np.pad(stretched, (0, expected_length - stretched.size))
    return stretched


def _stretch_pcm_bytes(pcm_bytes: bytes, speed: float) -> bytes:
    import numpy as np

    speed = _validate_speed(speed)
    if speed == _DEFAULT_SPEED or not pcm_bytes:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    stretched = _stretch_samples(samples, speed)
    clipped = np.clip(stretched, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


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
        dest.write_bytes(audio_bytes)

        # Evict stale cached state when re-registering the same id
        self._voice_states.pop(voice_id, None)
        self._custom_voice_files[voice_id] = dest

        # Pre-compute voice state (blocking, run in executor)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._get_voice_state, voice_id)

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
        if self._model is None:
            raise RuntimeError("TTSEngine not started")

        effective_speed = _validate_speed(self.speed if speed is None else speed)

        async with self._lock:
            loop = asyncio.get_running_loop()
            raw_chunks: list[bytes] = []

            # Ensure voice state is loaded
            voice_state = await loop.run_in_executor(
                None,
                self._get_voice_state,
                voice,
            )

            # Create the sync generator
            gen = self._model.generate_audio_stream(
                model_state=voice_state,
                text_to_generate=text,
                copy_state=True,
            )

            # Yield chunks by advancing the sync generator in executor
            while True:
                chunk_tensor = await loop.run_in_executor(
                    None,
                    functools.partial(next, gen, None),
                )
                if chunk_tensor is None:
                    break
                # Convert float tensor to int16 PCM bytes
                import numpy as np

                arr = chunk_tensor.numpy()
                pcm = np.clip(arr, -1.0, 1.0)
                pcm = (pcm * 32767).astype(np.int16)
                raw_chunk = pcm.tobytes()
                if effective_speed == _DEFAULT_SPEED:
                    yield raw_chunk
                    continue
                raw_chunks.append(raw_chunk)

            if effective_speed != _DEFAULT_SPEED:
                stretched = await loop.run_in_executor(
                    None,
                    _stretch_pcm_bytes,
                    b"".join(raw_chunks),
                    effective_speed,
                )
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

    async def start(self) -> None:
        self._engine = TTSEngine(
            voices_dir=self._voices_dir,
            default_voice=self._default_voice,
            speed=self._speed,
        )
        await self._engine.start()

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.stop()

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

    def router(self) -> APIRouter:
        docs_path = Path(__file__).resolve().parents[1] / "docs" / "server.md"
        if not docs_path.exists():
            docs_path = Path(__file__).resolve().parents[3] / "docs" / "server.md"
        try:
            import multipart  # noqa: F401
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
