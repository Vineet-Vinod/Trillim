"""Managed Pocket TTS engine for the TTS component."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
from pathlib import Path

from trillim.components.tts._limits import (
    MAX_PCM_CHUNK_BYTES,
    MAX_VOICE_STATE_BYTES,
    PROGRESS_TIMEOUT_SECONDS,
    TARGET_TTS_TOKENS,
    WORKER_KILL_AFTER_SECONDS,
)
from trillim.components.tts._validation import (
    dump_voice_state_safetensors_bytes,
    load_safe_voice_state_safetensors_bytes,
    validate_speed,
    validate_text,
    validate_voice_state_bytes,
)
from trillim.errors import ComponentLifecycleError, InvalidRequestError, ProgressTimeoutError


class TTSEngineError(RuntimeError):
    """Base class for TTS engine failures."""


class TTSEngineCrashedError(TTSEngineError):
    """Raised when synthesis fails and the engine recovers."""


class TTSEngineCatastrophicError(TTSEngineError, ComponentLifecycleError):
    """Raised when the engine cannot be recovered."""


_REQUEST_HEADER = struct.Struct(">I")
_RESPONSE_HEADER = struct.Struct(">cI")
# Custom voice states are base64 encoded inside JSON, so the frame limit must allow
# 4 output bytes for every 3 raw bytes plus slack for text and protocol fields.
_MAX_REQUEST_BYTES = ((MAX_VOICE_STATE_BYTES + 2) // 3) * 4 + 64 * 1024


class TTSEngine:
    """Own one long-lived Pocket TTS subprocess for segment synthesis."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start the Pocket TTS subprocess if it is not already running."""
        if self._process is not None and self._process.returncode is None:
            return
        self._process = await self._start_engine()

    async def stop(self) -> None:
        """Stop the Pocket TTS subprocess."""
        await self._stop_engine()

    async def recover(self) -> None:
        """Restart the Pocket TTS subprocess after a failed request."""
        await self._stop_engine()
        self._process = await self._start_engine()

    async def synthesize_segment(
        self,
        text: str,
        *,
        voice_state: str | bytes | bytearray | memoryview | dict,
        speed: float,
    ) -> bytes:
        """Synthesize one text segment to raw 16-bit PCM."""
        text = validate_text(text)
        speed = validate_speed(speed)
        request = _encode_synthesis_request(
            text=text,
            voice_state=voice_state,
            speed=speed,
        )

        process = self._process
        if process is None or process.returncode is not None:
            raise ComponentLifecycleError("TTS engine is not running")

        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None:
            await self._stop_engine()
            raise TTSEngineCatastrophicError("TTS engine pipes are unavailable")

        try:
            stdin.write(_REQUEST_HEADER.pack(len(request)))
            stdin.write(request)
            await asyncio.wait_for(stdin.drain(), timeout=WORKER_KILL_AFTER_SECONDS)
            kind, payload = await asyncio.wait_for(
                _read_response(stdout),
                timeout=PROGRESS_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            await self.recover()
            raise
        except TimeoutError as exc:
            await self.recover()
            raise ProgressTimeoutError(
                f"TTS chunk timed out after {PROGRESS_TIMEOUT_SECONDS} seconds"
            ) from exc
        except Exception as exc:
            await self.recover()
            raise TTSEngineCrashedError(
                "TTS engine crashed during synthesis and was recovered"
            ) from exc

        if kind == b"A":
            return payload
        if kind == b"E":
            await self.recover()
            raise TTSEngineCrashedError(
                payload.decode("utf-8", errors="replace")
                or "TTS engine failed during synthesis"
            )

        await self.recover()
        raise TTSEngineCrashedError(
            "TTS engine returned malformed output and was recovered"
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
                raise TTSEngineCatastrophicError("TTS engine stdout is unavailable")
            kind, payload = await asyncio.wait_for(
                _read_response(stdout),
                timeout=PROGRESS_TIMEOUT_SECONDS,
            )
            if kind == b"R":
                return process
            message = payload.decode("utf-8", errors="replace") or "TTS engine failed to start"
            raise TTSEngineCatastrophicError(message)
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
                request = json.dumps({"command": "stop"}).encode("utf-8")
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


def _encode_synthesis_request(
    *,
    text: str,
    voice_state: str | bytes | bytearray | memoryview | dict,
    speed: float,
) -> bytes:
    if isinstance(voice_state, str):
        state: dict[str, str] = {"kind": "predefined", "name": voice_state}
    else:
        if isinstance(voice_state, dict):
            voice_state = dump_voice_state_safetensors_bytes(voice_state)
        elif isinstance(voice_state, bytearray):
            voice_state = bytes(voice_state)
        elif isinstance(voice_state, memoryview):
            voice_state = voice_state.tobytes()
        elif not isinstance(voice_state, bytes):
            raise InvalidRequestError("voice_state must be a voice name or bytes")
        state = {
            "kind": "serialized",
            "data": base64.b64encode(validate_voice_state_bytes(voice_state)).decode("ascii"),
        }
    payload = json.dumps(
        {
            "command": "synthesize",
            "text": text,
            "voice_state": state,
            "speed": speed,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_REQUEST_BYTES:
        raise InvalidRequestError("TTS engine request is too large")
    return payload


async def _read_response(stream: asyncio.StreamReader) -> tuple[bytes, bytes]:
    header = await stream.readexactly(_RESPONSE_HEADER.size)
    kind, size = _RESPONSE_HEADER.unpack(header)
    if size > MAX_PCM_CHUNK_BYTES and kind == b"A":
        raise TTSEngineCrashedError("TTS engine produced oversized audio output")
    if size > _MAX_REQUEST_BYTES and kind != b"A":
        raise TTSEngineCrashedError("TTS engine produced oversized control output")
    return kind, await stream.readexactly(size)


def _write_response(kind: bytes, payload: bytes) -> None:
    sys.stdout.buffer.write(_RESPONSE_HEADER.pack(kind, len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _worker_main() -> int:
    try:
        from pocket_tts import TTSModel

        model = TTSModel.load_model()
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
                raise RuntimeError("TTS engine received malformed stdin input")
            (size,) = _REQUEST_HEADER.unpack(header)
            if size > _MAX_REQUEST_BYTES:
                raise RuntimeError("TTS engine request is too large")
            payload = stdin.read(size)
            if len(payload) != size:
                raise RuntimeError("TTS engine received malformed stdin input")
            request = json.loads(payload)
            if request.get("command") == "stop":
                return 0
            if request.get("command") != "synthesize":
                raise RuntimeError("TTS engine received unknown command")
            pcm = _synthesize_worker_request(model, request)
            _write_response(b"A", pcm)
        except Exception as exc:
            _write_response(b"E", _error_payload(exc))


def _synthesize_worker_request(model, request: object) -> bytes:
    if not isinstance(request, dict):
        raise RuntimeError("TTS engine received malformed request")
    text = request.get("text")
    voice_state = request.get("voice_state")
    if not isinstance(text, str) or not isinstance(voice_state, dict):
        raise RuntimeError("TTS engine received malformed request")
    state = _load_request_voice_state(model, voice_state)
    audio = model.generate_audio(
        model_state=state,
        text_to_generate=text,
        max_tokens=TARGET_TTS_TOKENS,
    )
    return _audio_tensor_to_pcm_bytes(audio)


def _load_request_voice_state(model, voice_state: dict):
    kind = voice_state.get("kind")
    if kind == "predefined" and isinstance(voice_state.get("name"), str):
        return model.get_state_for_audio_prompt(voice_state["name"])
    if kind == "serialized" and isinstance(voice_state.get("data"), str):
        state_bytes = base64.b64decode(voice_state["data"].encode("ascii"), validate=True)
        return load_safe_voice_state_safetensors_bytes(state_bytes)
    raise RuntimeError("TTS engine received malformed voice_state")


def _audio_tensor_to_pcm_bytes(audio) -> bytes:
    import torch

    tensor = torch.as_tensor(audio, dtype=torch.float32).detach().cpu().flatten()
    tensor = tensor.clamp_(-1.0, 1.0).mul_(32767.0).round().to(torch.int16)
    return tensor.numpy().tobytes()


def _error_payload(exc: Exception) -> bytes:
    return (str(exc) or type(exc).__name__).encode("utf-8", errors="replace")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
