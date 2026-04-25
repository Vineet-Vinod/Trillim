"""Managed faster-whisper engine for the STT."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
from pathlib import Path

from trillim.components.stt._config import DEFAULT_WORKER_CONFIG
from trillim.components.stt._limits import (
    MAX_WORKER_OUTPUT_BYTES,
    MAX_UPLOAD_BYTES,
    PCM_WIDTH_BYTES,
    TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS,
    WORKER_KILL_AFTER_SECONDS,
    STARTUP_TIMEOUT_SECONDS,
)
from trillim.errors import ComponentLifecycleError, InvalidRequestError, ProgressTimeoutError


class STTEngineError(RuntimeError):
    """Base class for STT engine failures."""


class STTEngineCrashedError(STTEngineError):
    """Raised when a transcription fails and the engine recovers."""


class STTEngineCatastrophicError(STTEngineError, ComponentLifecycleError):
    """Raised when the engine cannot be recovered."""


_REQUEST_HEADER = struct.Struct(">I")
_RESPONSE_HEADER = struct.Struct(">cI")
_MAX_REQUEST_BYTES = ((MAX_UPLOAD_BYTES + 2) // 3) * 4 + 64 * 1024


class STTEngine:
    """Own one faster-whisper subprocess for chunked transcription."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        self._process = await self._start_engine()

    async def stop(self) -> None:
        await self._stop_engine()

    async def recover(self) -> None:
        await self._stop_engine()
        self._process = await self._start_engine()

    async def transcribe(
        self,
        pcm: bytes,
        *,
        conditioning_text: str = "",
        language: str | None = None,
    ) -> str:
        pcm = self._validate_pcm(pcm)
        conditioning_text = self._validate_conditioning_text(conditioning_text)
        language = self._validate_language(language)
        process = self._process
        if process is None or process.returncode is not None:
            raise ComponentLifecycleError("STT engine is not running")

        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None:
            await self._stop_engine()
            raise STTEngineCatastrophicError("STT engine pipes are unavailable")

        request = _encode_transcription_request(
            pcm=pcm,
            conditioning_text=conditioning_text,
            language=language,
        )

        try:
            stdin.write(_REQUEST_HEADER.pack(len(request)))
            stdin.write(request)
            await asyncio.wait_for(stdin.drain(), timeout=WORKER_KILL_AFTER_SECONDS)
            kind, payload = await asyncio.wait_for(
                _read_response(stdout),
                timeout=TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            await self.recover()
            raise
        except TimeoutError as exc:
            await self.recover()
            raise ProgressTimeoutError(
                f"STT transcription timed out after {TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS} seconds"
            ) from exc
        except Exception as exc:
            await self.recover()
            raise STTEngineCrashedError(
                "STT engine crashed during transcription and was recovered"
            ) from exc

        if kind == b"T":
            return payload.decode("utf-8", errors="replace")
        if kind == b"E":
            await self.recover()
            raise STTEngineCrashedError(
                payload.decode("utf-8", errors="replace")
                or "STT engine failed during transcription"
            )

        await self.recover()
        raise STTEngineCrashedError(
            "STT engine returned malformed output and was recovered"
        )

    async def _start_engine(self) -> asyncio.subprocess.Process:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout = process.stdout
            if stdout is None:
                raise STTEngineCatastrophicError("STT engine stdout is unavailable")
            kind, payload = await asyncio.wait_for(
                _read_response(stdout),
                timeout=STARTUP_TIMEOUT_SECONDS,
            )
            if kind == b"R":
                return process
            message = payload.decode("utf-8", errors="replace") or "STT engine failed to start"
            raise STTEngineCatastrophicError(message)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise

    async def _stop_engine(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.returncode is not None:
            await process.wait()
            return

        stdin = process.stdin
        if stdin is not None:
            try:
                request = json.dumps({"command": "stop"}, separators=(",", ":")).encode("utf-8")
                stdin.write(_REQUEST_HEADER.pack(len(request)))
                stdin.write(request)
                await asyncio.wait_for(stdin.drain(), timeout=WORKER_KILL_AFTER_SECONDS)
            except Exception:
                pass

        try:
            await asyncio.wait_for(process.wait(), timeout=WORKER_KILL_AFTER_SECONDS)
            return
        except TimeoutError:
            pass

        process.kill()
        await process.wait()

    def _validate_pcm(self, pcm: bytes) -> bytes:
        if isinstance(pcm, bytearray):
            pcm = bytes(pcm)
        elif isinstance(pcm, memoryview):
            pcm = pcm.tobytes()
        elif not isinstance(pcm, bytes):
            raise InvalidRequestError("PCM audio must be bytes")
        if not pcm:
            raise InvalidRequestError("PCM audio must not be empty")
        if len(pcm) % PCM_WIDTH_BYTES != 0:
            raise InvalidRequestError("PCM audio must contain whole 16-bit samples")
        return pcm

    def _validate_conditioning_text(self, conditioning_text: str) -> str:
        if not isinstance(conditioning_text, str):
            raise InvalidRequestError("conditioning_text must be a string")
        return conditioning_text

    def _validate_language(self, language: str | None) -> str | None:
        if language is not None and not isinstance(language, str):
            raise InvalidRequestError("language must be a string")
        return language


def _encode_transcription_request(
    *,
    pcm: bytes,
    conditioning_text: str,
    language: str | None,
) -> bytes:
    payload = json.dumps(
        {
            "command": "transcribe",
            "pcm": base64.b64encode(pcm).decode("ascii"),
            "conditioning_text": conditioning_text,
            "language": language,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_REQUEST_BYTES:
        raise InvalidRequestError("STT engine request is too large")
    return payload


async def _read_response(stream: asyncio.StreamReader) -> tuple[bytes, bytes]:
    header = await stream.readexactly(_RESPONSE_HEADER.size)
    kind, size = _RESPONSE_HEADER.unpack(header)
    if size > MAX_WORKER_OUTPUT_BYTES:
        raise STTEngineCrashedError("STT engine produced oversized stdout")
    return kind, await stream.readexactly(size)


def _write_response(kind: bytes, payload: bytes) -> None:
    sys.stdout.buffer.write(_RESPONSE_HEADER.pack(kind, len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _worker_main() -> int:
    try:
        import faster_whisper
        import numpy as np

        model = faster_whisper.WhisperModel(
            DEFAULT_WORKER_CONFIG.model_name,
            device=DEFAULT_WORKER_CONFIG.device,
            compute_type=DEFAULT_WORKER_CONFIG.compute_type,
        )
        _write_response(b"R", b"")
    except Exception as exc:
        _write_response(b"E", _error_payload(exc))
        return 1

    stdin = sys.stdin.buffer
    while True:
        try:
            header = stdin.read(_REQUEST_HEADER.size)
            if not header:
                return 0
            if len(header) != _REQUEST_HEADER.size:
                raise RuntimeError("STT engine received malformed stdin input")
            (size,) = _REQUEST_HEADER.unpack(header)
            if size > _MAX_REQUEST_BYTES:
                raise RuntimeError("STT engine request is too large")
            payload = stdin.read(size)
            if len(payload) != size:
                raise RuntimeError("STT engine received malformed stdin input")
            request = json.loads(payload)

            command = request.get("command")
            if command == "stop":
                return 0
            if command != "transcribe":
                raise RuntimeError("STT engine received unknown command")

            pcm_b64 = request.get("pcm")
            conditioning_text = request.get("conditioning_text", "")
            language = request.get("language")
            if (
                not isinstance(pcm_b64, str)
                or not isinstance(conditioning_text, str)
                or (language is not None and not isinstance(language, str))
            ):
                raise RuntimeError("STT engine received malformed request")

            pcm = base64.b64decode(pcm_b64.encode("ascii"), validate=True)
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            segments, _info = model.transcribe(
                audio,
                language=language,
                initial_prompt=conditioning_text or None,
                condition_on_previous_text=False,
                without_timestamps=True,
                word_timestamps=False,
                vad_filter=False,
            )
            text = "".join(segment.text for segment in segments).strip()
            _write_response(b"T", text.encode("utf-8", errors="replace"))
        except Exception as exc:
            _write_response(b"E", _error_payload(exc))


def _error_payload(exc: Exception) -> bytes:
    return (str(exc) or type(exc).__name__).encode("utf-8", errors="replace")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
