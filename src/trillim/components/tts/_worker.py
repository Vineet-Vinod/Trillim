"""Subprocess worker helpers for the TTS component."""

from __future__ import annotations

import argparse
import asyncio
import io
import struct
import sys
from pathlib import Path

from trillim.components.tts._limits import (
    MAX_PCM_CHUNK_BYTES,
    MAX_VOICE_STATE_BYTES,
    MAX_WORKER_ERROR_BYTES,
    PROGRESS_TIMEOUT_SECONDS,
    TARGET_TTS_TOKENS,
    VOICE_STATE_BUILD_TIMEOUT_SECONDS,
    WORKER_KILL_AFTER_SECONDS,
)
from trillim.components.tts._validation import load_safe_voice_state_bytes
from trillim.errors import ProgressTimeoutError
from trillim.utils.formatting import human_size


class WorkerFailureError(RuntimeError):
    """Raised when the PocketTTS subprocess fails closed."""


class _WorkerStreamTooLargeError(RuntimeError):
    """Raised when one worker pipe exceeds its configured bound."""


_RESPONSE_HEADER = struct.Struct(">cI")
_REQUEST_HEADER = struct.Struct(">I")
_WORKER_BOOTSTRAP = (
    "from trillim.components.tts._worker import main; "
    "raise SystemExit(main())"
)
_VOICE_CLONE_AUTH_ERROR = """ValueError: We could not download the weights for the model with voice cloning, but you're trying to use voice cloning. Without voice cloning, you can use our catalog of voices ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']. If you want access to the model with voice cloning, go to https://huggingface.co/kyutai/pocket-tts and accept the terms, then make sure you're logged in locally with `uvx hf auth login`""".lower()


def create_session_worker(*, voice_kind: str, voice_reference: str) -> _PersistentSessionWorker:
    """Create one persistent worker client for a single TTS session."""
    return _PersistentSessionWorker(
        voice_kind=voice_kind,
        voice_reference=voice_reference,
    )


async def synthesize_segment(
    text: str,
    *,
    voice_kind: str,
    voice_reference: str,
) -> bytes:
    """Run one bounded TTS chunk in a child process and return raw PCM."""
    process = await asyncio.create_subprocess_exec(
        *_worker_command(
            "synthesize",
            text=text,
            voice_kind=voice_kind,
            voice_reference=voice_reference,
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await _collect_worker_output(
            process,
            stdout_limit=MAX_PCM_CHUNK_BYTES,
            timeout=PROGRESS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        await _stop_process(process)
        raise ProgressTimeoutError(
            f"TTS chunk timed out after {PROGRESS_TIMEOUT_SECONDS} seconds"
        ) from exc
    except _WorkerStreamTooLargeError as exc:
        await _stop_process(process)
        raise WorkerFailureError(str(exc)) from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    if process.returncode != 0:
        raise WorkerFailureError(_error_message(stderr))
    return stdout


class _PersistentSessionWorker:
    """One long-lived PocketTTS worker reused across one live TTS session."""

    def __init__(self, *, voice_kind: str, voice_reference: str) -> None:
        self._voice_kind = voice_kind
        self._voice_reference = voice_reference
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_chunks = bytearray()
        self._stderr_overflow: _WorkerStreamTooLargeError | None = None
        self._closed = False

    async def synthesize(self, text: str) -> bytes:
        """Synthesize one bounded segment through the persistent worker."""
        if self._closed:
            raise RuntimeError("TTS session worker is closed")
        async with self._lock:
            process = await self._ensure_process()
            try:
                await asyncio.wait_for(
                    self._write_request(process, text),
                    timeout=PROGRESS_TIMEOUT_SECONDS,
                )
                kind, payload = await asyncio.wait_for(
                    self._read_response(process),
                    timeout=PROGRESS_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                await self.close()
                raise ProgressTimeoutError(
                    f"TTS chunk timed out after {PROGRESS_TIMEOUT_SECONDS} seconds"
                ) from exc
            except asyncio.CancelledError:
                await self.close()
                raise
            except _WorkerStreamTooLargeError as exc:
                await self.close()
                raise WorkerFailureError(str(exc)) from exc
            except WorkerFailureError:
                await self.close()
                raise
            if kind != b"A":
                await self.close()
                raise WorkerFailureError("TTS worker produced malformed stdout output")
            return payload

    async def close(self) -> None:
        """Kill the persistent worker and release its pipes."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        stderr_task = self._stderr_task
        self._process = None
        self._stderr_task = None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            await _stop_process(process)
        if stderr_task is not None:
            await asyncio.gather(stderr_task, return_exceptions=True)

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is not None and process.returncode is None:
            self._raise_if_stderr_failed()
            return process
        process = await asyncio.create_subprocess_exec(
            *_worker_command(
                "session",
                voice_kind=self._voice_kind,
                voice_reference=self._voice_reference,
            ),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._process = process
        self._stderr_chunks = bytearray()
        self._stderr_overflow = None
        self._stderr_task = asyncio.create_task(self._drain_stderr(process))
        return process

    async def _write_request(
        self,
        process: asyncio.subprocess.Process,
        text: str,
    ) -> None:
        self._raise_if_stderr_failed()
        if process.stdin is None:
            raise WorkerFailureError(self._failure_message())
        payload = text.encode("utf-8")
        process.stdin.write(_REQUEST_HEADER.pack(len(payload)))
        process.stdin.write(payload)
        try:
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            await self._await_process_error(process)
            raise WorkerFailureError(self._failure_message()) from exc

    async def _read_response(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[bytes, bytes]:
        self._raise_if_stderr_failed()
        if process.stdout is None:
            raise WorkerFailureError(self._failure_message())
        try:
            header = await process.stdout.readexactly(_RESPONSE_HEADER.size)
            kind, size = _RESPONSE_HEADER.unpack(header)
            if size > MAX_PCM_CHUNK_BYTES:
                raise _WorkerStreamTooLargeError(
                    "TTS worker produced oversized stdout output"
                )
            payload = await process.stdout.readexactly(size)
        except asyncio.IncompleteReadError as exc:
            await self._await_process_error(process)
            raise WorkerFailureError(self._failure_message()) from exc
        self._raise_if_stderr_failed()
        if process.returncode not in (None, 0):
            await self._await_process_error(process)
            raise WorkerFailureError(self._failure_message())
        return kind, payload

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        while True:
            chunk = await process.stderr.read(64 * 1024)
            if not chunk:
                return
            if len(self._stderr_chunks) + len(chunk) > MAX_WORKER_ERROR_BYTES:
                self._stderr_overflow = _WorkerStreamTooLargeError(
                    "TTS worker produced oversized stderr output"
                )
                process.kill()
                return
            self._stderr_chunks.extend(chunk)

    def _raise_if_stderr_failed(self) -> None:
        if self._stderr_overflow is not None:
            raise self._stderr_overflow

    def _failure_message(self) -> str:
        self._raise_if_stderr_failed()
        return _error_message(bytes(self._stderr_chunks))

    async def _await_process_error(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            await process.wait()
        if self._stderr_task is not None:
            await asyncio.gather(self._stderr_task, return_exceptions=True)


async def build_voice_state(audio_path: str | Path) -> bytes:
    """Build one serialized custom-voice state in a child process."""
    process = await asyncio.create_subprocess_exec(
        *_worker_command("voice-state", audio_path=str(audio_path)),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await _collect_worker_output(
            process,
            stdout_limit=MAX_VOICE_STATE_BYTES,
            timeout=VOICE_STATE_BUILD_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        await _stop_process(process)
        raise ProgressTimeoutError(
            "TTS voice-state build timed out after "
            f"{VOICE_STATE_BUILD_TIMEOUT_SECONDS} seconds"
        ) from exc
    except _WorkerStreamTooLargeError as exc:
        await _stop_process(process)
        message = (
            _voice_state_too_large_message()
            if str(exc) == "TTS worker produced oversized stdout output"
            else str(exc)
        )
        raise WorkerFailureError(message) from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    if process.returncode != 0:
        raise WorkerFailureError(_error_message(stderr))
    return stdout


def is_voice_cloning_auth_error(message: str) -> bool:
    """Return whether one worker error is the explicit HF auth/terms failure."""
    text = message.lower()
    return _VOICE_CLONE_AUTH_ERROR in text


def _worker_command(command: str, **kwargs: str) -> tuple[str, ...]:
    args = [sys.executable, "-c", _WORKER_BOOTSTRAP, command]
    for key, value in kwargs.items():
        args.extend([f"--{key.replace('_', '-')}", value])
    return tuple(args)


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=WORKER_KILL_AFTER_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _error_message(stderr: bytes) -> str:
    message = stderr[:MAX_WORKER_ERROR_BYTES].decode("utf-8", errors="replace").strip()
    return message or "TTS worker failed"


def _voice_state_too_large_message() -> str:
    return (
        f"custom voice state exceeds the {human_size(MAX_VOICE_STATE_BYTES)} limit; "
        "use a shorter reference sample"
    )


async def _collect_worker_output(
    process: asyncio.subprocess.Process,
    *,
    stdout_limit: int,
    timeout: float | None = None,
) -> tuple[bytes, bytes]:
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(
        _read_bounded_stream(
            process.stdout,
            limit=stdout_limit,
            overflow_message="TTS worker produced oversized stdout output",
        )
    )
    stderr_task = asyncio.create_task(
        _read_bounded_stream(
            process.stderr,
            limit=MAX_WORKER_ERROR_BYTES,
            overflow_message="TTS worker produced oversized stderr output",
        )
    )
    wait_task = asyncio.create_task(process.wait())
    gather = asyncio.gather(wait_task, stdout_task, stderr_task)
    try:
        if timeout is None:
            _, stdout, stderr = await gather
        else:
            _, stdout, stderr = await asyncio.wait_for(gather, timeout=timeout)
    finally:
        await _finish_worker_tasks(wait_task, stdout_task, stderr_task)
    return stdout, stderr


async def _finish_worker_tasks(*tasks: asyncio.Task) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _read_bounded_stream(
    stream: asyncio.StreamReader,
    *,
    limit: int,
    overflow_message: str,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise _WorkerStreamTooLargeError(overflow_message)
        chunks.append(chunk)


def _synthesize_locally(
    text: str,
    *,
    voice_kind: str,
    voice_reference: str,
) -> bytes:
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    state = _load_worker_state(model, voice_kind=voice_kind, voice_reference=voice_reference)
    audio = model.generate_audio(
        model_state=state,
        text_to_generate=text,
        max_tokens=TARGET_TTS_TOKENS,
    )
    return _audio_tensor_to_pcm_bytes(audio)


def _build_voice_state_locally(audio_path: Path) -> bytes:
    import torch
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt(audio_path)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _load_worker_state(model, *, voice_kind: str, voice_reference: str):
    if voice_kind == "predefined":
        return model.get_state_for_audio_prompt(voice_reference)
    if voice_kind == "state_file":
        state = load_safe_voice_state_bytes(Path(voice_reference).read_bytes())
        _validate_state_file_voice_state(model, state)
        return state
    raise ValueError(f"unsupported voice kind: {voice_kind}")


def _validate_state_file_voice_state(model, state: dict) -> None:
    expected_keys_by_module = _bind_stateful_module_names(model)
    actual_module_names = set(state)
    expected_module_names = set(expected_keys_by_module)
    missing_modules = expected_module_names - actual_module_names
    if missing_modules:
        raise RuntimeError(
            "custom voice state is incompatible with the installed PocketTTS model: "
            f"missing module state for {_format_name_set(missing_modules)}"
        )
    unexpected_modules = actual_module_names - expected_module_names
    if unexpected_modules:
        raise RuntimeError(
            "custom voice state is incompatible with the installed PocketTTS model: "
            f"unexpected module state for {_format_name_set(unexpected_modules)}"
        )
    for module_name, expected_keys in expected_keys_by_module.items():
        module_state = state.get(module_name)
        if not isinstance(module_state, dict):
            raise RuntimeError(
                "custom voice state is incompatible with the installed PocketTTS model: "
                f"module state for {module_name!r} is malformed"
            )
        actual_keys = set(module_state)
        missing_keys = expected_keys - actual_keys
        if missing_keys:
            raise RuntimeError(
                "custom voice state is incompatible with the installed PocketTTS model: "
                f"module state for {module_name!r} is missing keys "
                f"{_format_name_set(missing_keys)}"
            )
        unexpected_keys = actual_keys - expected_keys
        if unexpected_keys:
            raise RuntimeError(
                "custom voice state is incompatible with the installed PocketTTS model: "
                f"module state for {module_name!r} contains unexpected keys "
                f"{_format_name_set(unexpected_keys)}"
            )


def _bind_stateful_module_names(model) -> dict[str, set[str]]:
    from pocket_tts.modules.stateful_module import StatefulModule

    flow_lm = getattr(model, "flow_lm", None)
    if flow_lm is None:
        raise RuntimeError(
            "custom voice state is incompatible with the installed PocketTTS model: "
            "model is missing flow_lm"
        )
    expected_keys_by_module: dict[str, set[str]] = {}
    for module_name, module in flow_lm.named_modules():
        if not isinstance(module, StatefulModule):
            continue
        module._module_absolute_name = module_name
        initial_state = module.init_state(batch_size=1, sequence_length=1)
        if not isinstance(initial_state, dict):
            raise RuntimeError(
                "custom voice state is incompatible with the installed PocketTTS model: "
                f"module {module_name!r} produced malformed state"
            )
        expected_keys_by_module[module_name] = set(initial_state)
    if not expected_keys_by_module:
        raise RuntimeError(
            "custom voice state is incompatible with the installed PocketTTS model: "
            "model has no stateful modules"
        )
    return expected_keys_by_module


def _format_name_set(names: set[str], *, limit: int = 4) -> str:
    ordered = sorted(names)
    shown = ", ".join(repr(name) for name in ordered[:limit])
    if len(ordered) > limit:
        shown = f"{shown}, ..."
    return shown


def _audio_tensor_to_pcm_bytes(audio) -> bytes:
    import torch

    tensor = torch.as_tensor(audio, dtype=torch.float32).detach().cpu().flatten()
    tensor = tensor.clamp_(-1.0, 1.0).mul_(32767.0).round().to(torch.int16)
    return tensor.numpy().tobytes()


def _run_session_worker(*, voice_kind: str, voice_reference: str) -> int:
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    state = _load_worker_state(
        model,
        voice_kind=voice_kind,
        voice_reference=voice_reference,
    )
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        header = stdin.read(_REQUEST_HEADER.size)
        if not header:
            return 0
        if len(header) != _REQUEST_HEADER.size:
            raise RuntimeError("TTS worker received malformed stdin input")
        (size,) = _REQUEST_HEADER.unpack(header)
        payload = stdin.read(size)
        if len(payload) != size:
            raise RuntimeError("TTS worker received malformed stdin input")
        text = payload.decode("utf-8")
        audio = model.generate_audio(
            model_state=state,
            text_to_generate=text,
            max_tokens=TARGET_TTS_TOKENS,
        )
        pcm = _audio_tensor_to_pcm_bytes(audio)
        stdout.write(_RESPONSE_HEADER.pack(b"A", len(pcm)))
        stdout.write(pcm)
        stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Run the TTS subprocess entrypoint."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    session_parser = subparsers.add_parser("session")
    session_parser.add_argument("--voice-kind", required=True)
    session_parser.add_argument("--voice-reference", required=True)

    synth_parser = subparsers.add_parser("synthesize")
    synth_parser.add_argument("--text", required=True)
    synth_parser.add_argument("--voice-kind", required=True)
    synth_parser.add_argument("--voice-reference", required=True)

    state_parser = subparsers.add_parser("voice-state")
    state_parser.add_argument("--audio-path", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "session":
            return _run_session_worker(
                voice_kind=args.voice_kind,
                voice_reference=args.voice_reference,
            )
        if args.command == "synthesize":
            output = _synthesize_locally(
                args.text,
                voice_kind=args.voice_kind,
                voice_reference=args.voice_reference,
            )
        else:
            output = _build_voice_state_locally(Path(args.audio_path))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
