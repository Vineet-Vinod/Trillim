"""Managed faster-whisper engine for the STT."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

from trillim.components.stt._config import DEFAULT_WORKER_CONFIG
from trillim.components.stt._limits import (
    MAX_WORKER_OUTPUT_BYTES,
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

        request = json.dumps(
            {
                "pcm": base64.b64encode(pcm).decode("ascii"),
                "conditioning_text": conditioning_text,
                "language": language,
            }
        ).encode("utf-8") + b"\n"

        try:
            stdin.write(request)
            await asyncio.wait_for(stdin.drain(), timeout=WORKER_KILL_AFTER_SECONDS)
            response_line = await asyncio.wait_for(
                stdout.readline(),
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

        if not response_line:
            await self.recover()
            raise STTEngineCrashedError(
                "STT engine exited during transcription and was recovered"
            )
        if len(response_line) > MAX_WORKER_OUTPUT_BYTES:
            await self.recover()
            raise STTEngineCrashedError("STT engine produced oversized stdout and was recovered")

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            await self.recover()
            raise STTEngineCrashedError(
                "STT engine returned malformed output and was recovered"
            ) from exc

        if response.get("status") == "text" and isinstance(response.get("text"), str):
            return response["text"]

        if response.get("status") == "error" and isinstance(response.get("message"), str):
            await self.recover()
            raise STTEngineCrashedError(response["message"])

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
            response_line = await asyncio.wait_for(
                stdout.readline(),
                timeout=STARTUP_TIMEOUT_SECONDS,
            )
            if not response_line:
                raise STTEngineCatastrophicError("STT engine failed to start")
            if len(response_line) > MAX_WORKER_OUTPUT_BYTES:
                raise STTEngineCatastrophicError("STT engine produced oversized stdout")
            response = json.loads(response_line)
            if response.get("status") != "ready":
                raise STTEngineCatastrophicError(
                    str(response.get("message", "STT engine failed to start"))
                )
            return process
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
                stdin.write(b'{"command":"stop"}\n')
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
        if len(pcm) % 2 != 0:
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


def _worker_main() -> int:
    def write(payload: dict[str, str]) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    try:
        import faster_whisper
        import numpy as np

        model = faster_whisper.WhisperModel(
            DEFAULT_WORKER_CONFIG.model_name,
            device=DEFAULT_WORKER_CONFIG.device,
            compute_type=DEFAULT_WORKER_CONFIG.compute_type,
        )
        write({"status": "ready"})
    except Exception as exc:
        write({"status": "startup_error", "message": str(exc) or type(exc).__name__})
        return 1

    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            write({"status": "error", "message": "Malformed worker request"})
            continue

        if request.get("command") == "stop":
            return 0

        pcm_b64 = request.get("pcm")
        conditioning_text = request.get("conditioning_text", "")
        language = request.get("language")
        if (
            not isinstance(pcm_b64, str)
            or not isinstance(conditioning_text, str)
            or (language is not None and not isinstance(language, str))
        ):
            write({"status": "error", "message": "Malformed worker request"})
            continue

        try:
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
        except Exception as exc:
            write({"status": "error", "message": str(exc) or type(exc).__name__})
            continue

        write({"status": "text", "text": text})

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
