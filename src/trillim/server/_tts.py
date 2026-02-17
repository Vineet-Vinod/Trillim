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

    def __init__(self, voices_dir: Path | None = None):
        self._model = None
        self._voice_states: dict[str, dict] = {}
        self._custom_voice_files: dict[str, Path] = {}  # voice_id -> WAV path
        self._voices_dir = voices_dir
        self._lock = asyncio.Lock()
        self.sample_rate: int = 24000

    async def start(self) -> None:
        """Load the TTS model, default voice state, and discover saved voices."""
        loop = asyncio.get_running_loop()
        self._model, default_state = await loop.run_in_executor(
            None,
            self._load,
        )
        self.sample_rate = self._model.sample_rate
        self._voice_states[self.DEFAULT_VOICE] = default_state

        # Discover previously-saved custom voices (states loaded lazily)
        if self._voices_dir is not None:
            self._voices_dir.mkdir(parents=True, exist_ok=True)
            for wav in sorted(self._voices_dir.glob("*.wav")):
                voice_id = wav.stem
                if voice_id not in PREDEFINED_VOICES:
                    self._custom_voice_files[voice_id] = wav

    def _load(self):
        from pocket_tts import TTSModel

        model = TTSModel.load_model()
        model.eval()
        state = model.get_state_for_audio_prompt(self.DEFAULT_VOICE)
        return model, state

    async def stop(self) -> None:
        self._model = None
        self._voice_states.clear()
        self._custom_voice_files.clear()

    def _get_voice_state(self, voice: str | None) -> dict:
        voice = voice or self.DEFAULT_VOICE
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

    # ----- Synthesis -----

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Yield PCM int16 audio chunks for the given text."""
        if self._model is None:
            raise RuntimeError("TTSEngine not started")

        async with self._lock:
            loop = asyncio.get_running_loop()

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
                yield pcm.tobytes()

    async def synthesize_full(self, text: str, voice: str | None = None) -> bytes:
        """Synthesize text and return a complete WAV file as bytes."""
        chunks = []
        async for chunk in self.synthesize_stream(text, voice):
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

        from trillim.server import LLM, TTS, SentenceChunker

        chunker = SentenceChunker()
        for token_text in llm_stream:
            for sentence in chunker.feed(token_text):
                await tts.engine.synthesize_stream(sentence)
        remaining = chunker.flush()
        if remaining:
            await tts.engine.synthesize_stream(remaining)
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

    def __init__(self, voices_dir: str | Path = _DEFAULT_VOICES_DIR):
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._engine = None

    async def start(self) -> None:
        self._engine = TTSEngine(voices_dir=self._voices_dir)
        await self._engine.start()

    async def stop(self) -> None:
        if self._engine is not None:
            await self._engine.stop()

    @property
    def engine(self):
        return self._engine

    def router(self) -> APIRouter:
        r = APIRouter()
        tts = self

        @r.get("/v1/voices")
        async def list_voices():
            if tts._engine is None:
                raise HTTPException(status_code=503, detail="TTS engine not started")
            return VoiceListResponse(
                voices=[VoiceInfo(**v) for v in tts._engine.list_voices()],
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
                await tts._engine.register_voice(voice_id, audio_bytes)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return VoiceCreateResponse(voice_id=voice_id, status="created")

        @r.delete("/v1/voices/{voice_id}")
        async def delete_voice(voice_id: str):
            if tts._engine is None:
                raise HTTPException(status_code=503, detail="TTS engine not started")
            try:
                await tts._engine.delete_voice(voice_id)
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

            if req.response_format == "pcm":
                return StreamingResponse(
                    tts._engine.synthesize_stream(req.input, voice=req.voice),
                    media_type="audio/pcm",
                )

            async def _wav_stream():
                yield wav_header(tts._engine.sample_rate)
                async for chunk in tts._engine.synthesize_stream(
                    req.input,
                    voice=req.voice,
                ):
                    yield chunk

            return StreamingResponse(_wav_stream(), media_type="audio/wav")

        return r
