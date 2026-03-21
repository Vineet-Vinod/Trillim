"""Subprocess worker protocol for Phase 4 STT."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from trillim.components.stt._config import DEFAULT_WORKER_CONFIG
from trillim.components.stt._limits import (
    TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS,
    WORKER_KILL_AFTER_SECONDS,
)
from trillim.errors import ProgressTimeoutError


class WorkerFailureError(RuntimeError):
    """Raised when the STT subprocess fails closed."""


async def transcribe_owned_audio_file(
    audio_path: str | Path,
    *,
    language: str | None,
) -> str:
    """Run one STT request in a subprocess and return the final transcript."""
    process = await asyncio.create_subprocess_exec(
        *_worker_command(audio_path, language=language),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        await _stop_process(process)
        raise ProgressTimeoutError(
            f"STT transcription timed out after {TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS} seconds"
        ) from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    if process.returncode != 0:
        raise WorkerFailureError("STT worker failed")
    return _parse_worker_output(stdout)


def _worker_command(audio_path: str | Path, *, language: str | None) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "trillim.components.stt._worker",
        "--audio-path",
        str(audio_path),
    ]
    if language is not None:
        command.extend(["--language", language])
    return tuple(command)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=WORKER_KILL_AFTER_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _parse_worker_output(stdout: bytes) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise WorkerFailureError("STT worker produced malformed output") from exc
    if not isinstance(payload, dict):
        raise WorkerFailureError("STT worker produced malformed output")
    text = payload.get("text")
    if not isinstance(text, str):
        raise WorkerFailureError("STT worker produced malformed output")
    return text


def _transcribe_locally(audio_path: Path, *, language: str | None) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        DEFAULT_WORKER_CONFIG.model_name,
        device=DEFAULT_WORKER_CONFIG.device,
        compute_type=DEFAULT_WORKER_CONFIG.compute_type,
    )
    segments, _info = model.transcribe(str(audio_path), language=language)
    return "".join(segment.text for segment in segments).strip()


def main(argv: list[str] | None = None) -> int:
    """Run the subprocess worker entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--language", default=None)
    args = parser.parse_args(argv)
    try:
        text = _transcribe_locally(Path(args.audio_path), language=args.language)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps({"text": text}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
