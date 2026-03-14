# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Whisper component — speech-to-text using faster-whisper."""

import asyncio
import functools
import io
import struct
import wave
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from trillim._timeouts import run_with_timeout

from ._component import Component
from ._models import TranscriptionResponse

_MAX_TRANSCRIPTION_UPLOAD_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# WhisperEngine
# ---------------------------------------------------------------------------


class WhisperEngine:
    """Async wrapper around faster-whisper for speech-to-text."""

    def __init__(
        self,
        model_size: str = "base.en",
        compute_type: str = "int8",
        cpu_threads: int = 2,
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Load the Whisper model (blocking I/O run in executor)."""
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load)

    def _load(self):
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as e:
            docs_path = Path(__file__).resolve().parents[1] / "docs" / "server.md"
            if not docs_path.exists():
                docs_path = Path(__file__).resolve().parents[3] / "docs" / "server.md"
            raise RuntimeError(
                "Voice support is optional and requires the 'voice' extra. "
                "Install with uv or pip using 'trillim[voice]'. "
                f"Docs: {docs_path} (section: Voice Optional Dependencies)"
            ) from e

        return WhisperModel(
            self.model_size,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    async def stop(self) -> None:
        self._model = None

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """Transcribe audio bytes to text. Returns the full transcription."""
        if self._model is None:
            raise RuntimeError("WhisperEngine not started")

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(self._transcribe_sync, io.BytesIO(audio_bytes), language),
            )

    async def transcribe_path(
        self,
        path: str | Path,
        language: str | None = None,
    ) -> str:
        """Transcribe an audio file directly from disk."""
        if self._model is None:
            raise RuntimeError("WhisperEngine not started")

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                functools.partial(self._transcribe_sync, str(path), language),
            )

    def _transcribe_sync(self, audio_input, language: str | None) -> str:
        segments, _ = self._model.transcribe(
            audio_input,
            language=language,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments)


# ---------------------------------------------------------------------------
# Whisper component
# ---------------------------------------------------------------------------


class Whisper(Component):
    """Speech-to-text component using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base.en",
        compute_type: str = "int8",
        cpu_threads: int = 2,
    ):
        self._model_size = model_size
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._engine = None

    async def start(self) -> None:
        self._engine = WhisperEngine(
            model_size=self._model_size,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
        )
        await self._engine.start()

    async def stop(self) -> None:
        if self._engine is not None:
            engine = self._engine
            try:
                await engine.stop()
            finally:
                self._engine = None

    @property
    def engine(self):
        return self._engine

    def _require_started(self) -> WhisperEngine:
        if self._engine is None:
            raise RuntimeError("Whisper not started")
        return self._engine

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Transcribe supported audio bytes to text."""
        engine = self._require_started()
        return await run_with_timeout(
            engine.transcribe(audio_bytes, language=language),
            timeout,
            "Whisper transcription",
        )

    async def transcribe_wav(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Transcribe a WAV file directly from disk."""
        engine = self._require_started()
        return await run_with_timeout(
            engine.transcribe_path(path, language=language),
            timeout,
            "Whisper transcription",
        )

    async def transcribe_array(
        self,
        samples,
        *,
        sample_rate: int,
        channel_axis: int | None = None,
        language: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Encode an array-like audio buffer as WAV and transcribe it."""
        audio_bytes = _wav_bytes_from_array(
            samples,
            sample_rate,
            channel_axis=channel_axis,
        )
        return await self.transcribe_bytes(
            audio_bytes,
            language=language,
            timeout=timeout,
        )

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
        whisper = self

        @r.post("/v1/audio/transcriptions")
        async def transcriptions(
            file: UploadFile = File(...),
            model: str = Form("whisper-1"),
            language: str | None = Form(None),
            response_format: str = Form("json"),
        ):
            if whisper._engine is None:
                raise HTTPException(
                    status_code=503, detail="Whisper engine not started"
                )
            audio_bytes = await file.read(_MAX_TRANSCRIPTION_UPLOAD_BYTES + 1)
            if len(audio_bytes) > _MAX_TRANSCRIPTION_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Upload exceeds 10 MB limit")
            text = await whisper.transcribe_bytes(audio_bytes, language=language)
            if response_format == "text":
                return StreamingResponse(iter([text]), media_type="text/plain")
            return TranscriptionResponse(text=text)

        return r


def _wav_bytes_from_array(
    samples,
    sample_rate: int,
    *,
    channel_axis: int | None = None,
) -> bytes:
    """Convert an array-like audio buffer to mono 16-bit WAV bytes."""
    if sample_rate < 1:
        raise ValueError("sample_rate must be >= 1")

    mono = _coerce_mono_samples(samples, channel_axis=channel_axis)
    pcm = b"".join(struct.pack("<h", _float_to_int16(sample)) for sample in mono)

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()


def _coerce_mono_samples(samples, *, channel_axis: int | None = None) -> list[float]:
    if isinstance(samples, (bytes, bytearray, memoryview)):
        raise TypeError("samples must be an array-like sequence, not raw bytes")
    if hasattr(samples, "tolist"):
        scale_hint = _infer_scale_hint(samples)
        zero_point = _infer_zero_point(samples)
        samples = samples.tolist()
    else:
        if not isinstance(samples, (list, tuple)):
            samples = list(samples)
        scale_hint = _infer_scale_hint(samples)
        zero_point = _infer_zero_point(samples)

    if not samples:
        raise ValueError("samples must not be empty")

    first = samples[0]
    if _is_sequence(first):
        return _collapse_channels(
            samples,
            channel_axis=channel_axis,
            scale_hint=scale_hint,
            zero_point=zero_point,
        )
    return [
        _normalize_scalar(sample, scale_hint=scale_hint, zero_point=zero_point)
        for sample in samples
    ]


def _collapse_channels(
    samples,
    *,
    channel_axis: int | None = None,
    scale_hint: float | None = None,
    zero_point: float = 0.0,
) -> list[float]:
    rows = [list(row) if not hasattr(row, "tolist") else row.tolist() for row in samples]
    if not rows or not rows[0]:
        raise ValueError("samples must not be empty")
    row_lengths = {len(row) for row in rows}
    if len(row_lengths) != 1:
        raise ValueError("multichannel samples must have a consistent shape")
    if channel_axis not in (None, 0, 1):
        raise ValueError("channel_axis must be None, 0, or 1")

    if scale_hint is None:
        scale_hint = _infer_scale_hint(samples)
        zero_point = _infer_zero_point(samples)

    if channel_axis == 0:
        channels = rows
        frame_count = len(rows[0])
        return [
            sum(
                _normalize_scalar(
                    channel[i],
                    scale_hint=scale_hint,
                    zero_point=zero_point,
                )
                for channel in channels
            )
            / len(channels)
            for i in range(frame_count)
        ]

    # Default 2D layout is frames-first: (num_samples, channels).
    return [
        sum(
            _normalize_scalar(
                value,
                scale_hint=scale_hint,
                zero_point=zero_point,
            )
            for value in frame
        )
        / len(frame)
        for frame in rows
    ]


def _infer_scale_hint(samples) -> float | None:
    dtype = getattr(samples, "dtype", None)
    kind = getattr(dtype, "kind", None)
    itemsize = getattr(dtype, "itemsize", None)
    if kind == "i" and itemsize:
        return float(2 ** (8 * itemsize - 1))
    if kind == "u" and itemsize:
        return float(2 ** (8 * itemsize - 1))

    max_abs = 0.0
    for value in _flatten(samples):
        if isinstance(value, bool):
            continue
        try:
            max_abs = max(max_abs, abs(float(value)))
        except (TypeError, ValueError):
            raise TypeError(f"Unsupported sample value: {value!r}") from None

    if max_abs <= 1.0:
        return None
    if max_abs <= 32768:
        return 32768.0
    if max_abs <= 8388608:
        return 8388608.0
    if max_abs <= 2147483648:
        return 2147483648.0
    return max_abs


def _infer_zero_point(samples) -> float:
    dtype = getattr(samples, "dtype", None)
    kind = getattr(dtype, "kind", None)
    itemsize = getattr(dtype, "itemsize", None)
    if kind == "u" and itemsize:
        return float(2 ** (8 * itemsize - 1))
    return 0.0


def _flatten(samples):
    if isinstance(samples, (list, tuple)):
        for value in samples:
            if _is_sequence(value):
                yield from _flatten(value)
            else:
                yield value
        return

    if hasattr(samples, "tolist"):
        yield from _flatten(samples.tolist())
        return

    yield samples


def _normalize_scalar(
    value,
    *,
    scale_hint: float | None,
    zero_point: float = 0.0,
) -> float:
    if isinstance(value, bool):
        value = int(value)
    sample = float(value)
    if scale_hint is not None:
        sample = (sample - zero_point) / scale_hint
    return max(-1.0, min(1.0, sample))


def _float_to_int16(sample: float) -> int:
    return max(-32768, min(32767, int(round(sample * 32767.0))))


def _is_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) or hasattr(value, "tolist")
