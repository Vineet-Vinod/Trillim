# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Whisper component — speech-to-text using faster-whisper."""

import asyncio
import functools
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ._component import Component
from ._models import TranscriptionResponse


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
        from faster_whisper import WhisperModel

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
                functools.partial(self._transcribe_sync, audio_bytes, language),
            )

    def _transcribe_sync(self, audio_bytes: bytes, language: str | None) -> str:
        audio_file = io.BytesIO(audio_bytes)
        segments, _ = self._model.transcribe(
            audio_file,
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
            await self._engine.stop()

    @property
    def engine(self):
        return self._engine

    def router(self) -> APIRouter:
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
            audio_bytes = await file.read(8 * 1024 * 1024 + 1)
            if len(audio_bytes) > 8 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Upload exceeds 8 MB limit")
            text = await whisper._engine.transcribe(audio_bytes, language=language)
            if response_format == "text":
                return StreamingResponse(iter([text]), media_type="text/plain")
            return TranscriptionResponse(text=text)

        return r
