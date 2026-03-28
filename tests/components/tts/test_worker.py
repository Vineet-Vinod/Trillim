"""Tests for the TTS worker subprocess helpers."""

from __future__ import annotations

import asyncio
import io
import os
import runpy
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
import trillim.components.tts._worker as tts_worker_module

from trillim.components.tts._validation import load_safe_voice_state_bytes
from trillim.errors import ProgressTimeoutError
from trillim.components.tts._worker import (
    WorkerFailureError,
    _VOICE_CLONE_AUTH_ERROR,
    _REQUEST_HEADER,
    _RESPONSE_HEADER,
    _audio_tensor_to_pcm_bytes,
    _collect_worker_output,
    _error_message,
    _load_worker_state,
    _run_session_worker,
    _stop_process,
    build_voice_state,
    create_session_worker,
    is_voice_cloning_auth_error,
    main,
    synthesize_segment,
)
from tests.components.tts.support import prepend_pythonpath, write_fake_pocket_tts_package


class _UnsafeState:
    pass


def _write_counting_fake_pocket_tts_package(root: Path, counter_path: Path) -> None:
    write_fake_pocket_tts_package(root)
    (root / "pocket_tts" / "__init__.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                f"_COUNTER_PATH = Path({str(counter_path)!r})",
                "",
                "class _Model:",
                "    def get_state_for_audio_prompt(self, audio_conditioning):",
                "        text = str(audio_conditioning)",
                "        if text in {'alba', 'marius'}:",
                "            return {'prompt': text}",
                "        if str(audio_conditioning).endswith('.state'):",
                "            return {'state': 'preloaded'}",
                "        return {'prompt': Path(audio_conditioning).read_text(encoding='latin-1')}",
                "",
                "    def generate_audio(self, model_state, text_to_generate, max_tokens=20):",
                "        del model_state, text_to_generate, max_tokens",
                "        return [0.0, 0.25, -0.25]",
                "",
                "class TTSModel:",
                "    @staticmethod",
                "    def load_model():",
                "        count = 0",
                "        if _COUNTER_PATH.exists():",
                "            count = int(_COUNTER_PATH.read_text(encoding='utf-8') or '0')",
                "        _COUNTER_PATH.write_text(str(count + 1), encoding='utf-8')",
                "        return _Model()",
            ]
        ),
        encoding="utf-8",
    )


class TTSWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        write_fake_pocket_tts_package(self.root)

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_synthesize_segment_returns_pcm_bytes(self):
        with patch.dict(os.environ, prepend_pythonpath(self.root)):
            pcm = await synthesize_segment(
                "tiny prompt",
                voice_kind="predefined",
                voice_reference="alba",
            )
        self.assertEqual(len(pcm), 4)

    async def test_build_voice_state_returns_serialized_bytes(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")
        with patch.dict(os.environ, prepend_pythonpath(self.root)):
            state_bytes = await build_voice_state(audio_path)
        loaded = load_safe_voice_state_bytes(state_bytes)
        self.assertEqual(loaded["prompt"], "voice")

    async def test_persistent_session_worker_reuses_loaded_model_across_segments(self):
        counter_path = self.root / "loads.txt"
        _write_counting_fake_pocket_tts_package(self.root, counter_path)
        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        try:
            with patch.dict(os.environ, prepend_pythonpath(self.root)):
                first = await worker.synthesize("first chunk")
                second = await worker.synthesize("second chunk")
        finally:
            await worker.close()
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(counter_path.read_text(encoding="utf-8"), "1")

    async def test_worker_failures_surface_cleanly(self):
        deny_path = self.root / "deny"
        deny_path.write_text("x", encoding="latin-1")
        with patch.dict(os.environ, prepend_pythonpath(self.root)):
            with self.assertRaises(WorkerFailureError):
                await build_voice_state(deny_path)
            with self.assertRaises(WorkerFailureError):
                await synthesize_segment(
                    "boom",
                    voice_kind="predefined",
                    voice_reference="alba",
                )

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        try:
            with patch.dict(os.environ, prepend_pythonpath(self.root)):
                with self.assertRaises(WorkerFailureError):
                    await worker.synthesize("boom")
        finally:
            await worker.close()

    def test_audio_tensor_to_pcm_bytes_and_auth_error_detection(self):
        self.assertEqual(len(_audio_tensor_to_pcm_bytes([0.0, 0.5])), 4)
        self.assertTrue(is_voice_cloning_auth_error(_VOICE_CLONE_AUTH_ERROR.upper()))
        self.assertFalse(is_voice_cloning_auth_error("not an auth error"))

    def test_load_worker_state_rejects_unsafe_serialized_state(self):
        good_path = self.root / "voice.state"
        torch.save({"prompt": "voice"}, good_path)
        self.assertEqual(
            _load_worker_state(
                SimpleNamespace(),
                voice_kind="state_file",
                voice_reference=str(good_path),
            )["prompt"],
            "voice",
        )

        bad_path = self.root / "bad.state"
        torch.save({"bad": _UnsafeState()}, bad_path)
        with self.assertRaisesRegex(Exception, "voice state is malformed"):
            _load_worker_state(
                SimpleNamespace(),
                voice_kind="state_file",
                voice_reference=str(bad_path),
            )

        with self.assertRaisesRegex(ValueError, "unsupported voice kind"):
            _load_worker_state(
                SimpleNamespace(),
                voice_kind="unknown",
                voice_reference="ignored",
            )

    async def test_worker_helpers_bound_stdout_and_stderr(self):
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 9)",
            ),
        ), patch("trillim.components.tts._worker.MAX_PCM_CHUNK_BYTES", 8):
            with self.assertRaisesRegex(WorkerFailureError, "oversized stdout"):
                await synthesize_segment(
                    "tiny prompt",
                    voice_kind="predefined",
                    voice_reference="alba",
                )

        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('x' * 33); sys.exit(1)",
            ),
        ), patch("trillim.components.tts._worker.MAX_WORKER_ERROR_BYTES", 32):
            with self.assertRaisesRegex(WorkerFailureError, "oversized stderr"):
                await build_voice_state(self.root / "voice.txt")

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        try:
            with patch(
                "trillim.components.tts._worker._worker_command",
                return_value=(
                    sys.executable,
                    "-c",
                    "import sys, struct, time; "
                    "size = struct.unpack('>I', sys.stdin.buffer.read(4))[0]; "
                    "sys.stdin.buffer.read(size); "
                    "sys.stderr.write('x' * 33); "
                    "sys.stderr.flush(); "
                    "time.sleep(1)",
                ),
            ), patch("trillim.components.tts._worker.MAX_WORKER_ERROR_BYTES", 32):
                with self.assertRaisesRegex(WorkerFailureError, "oversized stderr"):
                    await worker.synthesize("hello")
        finally:
            await worker.close()

    async def test_build_voice_state_times_out(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import time; time.sleep(1)",
            ),
        ), patch("trillim.components.tts._worker.VOICE_STATE_BUILD_TIMEOUT_SECONDS", 0.01):
            with self.assertRaisesRegex(ProgressTimeoutError, "voice-state build timed out"):
                await build_voice_state(audio_path)

    async def test_build_voice_state_reports_voice_state_size_limit_clearly(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 9)",
            ),
        ), patch("trillim.components.tts._worker.MAX_VOICE_STATE_BYTES", 8):
            with self.assertRaisesRegex(
                WorkerFailureError,
                "custom voice state exceeds the 8 B limit; use a shorter reference sample",
            ):
                await build_voice_state(audio_path)

    async def test_synthesize_segment_timeout_and_nonzero_exit_surface_worker_failures(self):
        process = SimpleNamespace(returncode=None)
        with patch(
            "trillim.components.tts._worker.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ), patch(
            "trillim.components.tts._worker._collect_worker_output",
            new=AsyncMock(side_effect=asyncio.TimeoutError()),
        ), patch(
            "trillim.components.tts._worker._stop_process",
            new=AsyncMock(),
        ) as stop_process:
            with self.assertRaisesRegex(ProgressTimeoutError, "chunk timed out"):
                await synthesize_segment(
                    "tiny prompt",
                    voice_kind="predefined",
                    voice_reference="alba",
                )
        stop_process.assert_awaited_once_with(process)

        process = SimpleNamespace(returncode=1)
        with patch(
            "trillim.components.tts._worker.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ), patch(
            "trillim.components.tts._worker._collect_worker_output",
            new=AsyncMock(return_value=(b"", b"worker boom")),
        ):
            with self.assertRaisesRegex(WorkerFailureError, "worker boom"):
                await synthesize_segment(
                    "tiny prompt",
                    voice_kind="predefined",
                    voice_reference="alba",
                )

    async def test_persistent_session_worker_times_out_and_rejects_malformed_stdout(self):
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import time; time.sleep(1)",
            ),
        ), patch("trillim.components.tts._worker.PROGRESS_TIMEOUT_SECONDS", 0.01):
            worker = create_session_worker(
                voice_kind="predefined",
                voice_reference="alba",
            )
            try:
                with self.assertRaisesRegex(ProgressTimeoutError, "chunk timed out"):
                    await worker.synthesize("hello")
            finally:
                await worker.close()

    async def test_persistent_worker_internal_error_paths_cover_closed_cancelled_and_pipe_failures(self):
        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        await worker.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await worker.synthesize("hello")

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(close=lambda: None))

        async def cancelled_write(_process, _text):
            raise asyncio.CancelledError()

        worker._ensure_process = AsyncMock(return_value=process)
        worker._write_request = AsyncMock(side_effect=cancelled_write)
        worker.close = AsyncMock()
        with self.assertRaises(asyncio.CancelledError):
            await worker.synthesize("hello")
        worker.close.assert_awaited()

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        worker._stderr_task = None
        with self.assertRaises(WorkerFailureError):
            await worker._write_request(SimpleNamespace(stdin=None), "hello")

        class _BrokenWriter:
            def __init__(self) -> None:
                self.writes: list[bytes] = []

            def write(self, data: bytes) -> None:
                self.writes.append(data)

            async def drain(self) -> None:
                raise BrokenPipeError("boom")

        broken_stdin = _BrokenWriter()
        process = SimpleNamespace(
            stdin=broken_stdin,
            returncode=1,
            wait=AsyncMock(return_value=1),
        )
        worker._stderr_chunks = bytearray(b"stderr boom")
        worker._stderr_task = None
        with self.assertRaisesRegex(WorkerFailureError, "stderr boom"):
            await worker._write_request(process, "hello")

        worker._stderr_chunks = bytearray(b"stdout gone")
        with self.assertRaisesRegex(WorkerFailureError, "stdout gone"):
            await worker._read_response(SimpleNamespace(stdout=None))

        header = _RESPONSE_HEADER.pack(b"A", 0)

        class _Reader:
            async def readexactly(self, size: int) -> bytes:
                return header if size == _RESPONSE_HEADER.size else b""

        process = SimpleNamespace(stdout=_Reader(), returncode=1, wait=AsyncMock(return_value=1))
        worker._stderr_chunks = bytearray(b"response boom")
        worker._stderr_task = None
        with self.assertRaisesRegex(WorkerFailureError, "response boom"):
            await worker._read_response(process)

        wait_process = SimpleNamespace(returncode=None, wait=AsyncMock(return_value=1))
        worker._stderr_task = None
        await worker._await_process_error(wait_process)
        wait_process.wait.assert_awaited_once()

        malformed_script = (
            "import sys, struct; "
            "header = sys.stdin.buffer.read(4); "
            "size = struct.unpack('>I', header)[0]; "
            "sys.stdin.buffer.read(size); "
            "sys.stdout.buffer.write(struct.pack('>cI', b'B', 0)); "
            "sys.stdout.buffer.flush()"
        )
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(sys.executable, "-c", malformed_script),
        ):
            worker = create_session_worker(
                voice_kind="predefined",
                voice_reference="alba",
            )
            try:
                with self.assertRaisesRegex(WorkerFailureError, "malformed stdout"):
                    await worker.synthesize("hello")
            finally:
                await worker.close()

    async def test_worker_helper_paths_cover_cancellation_and_process_shutdown(self):
        process = SimpleNamespace(returncode=None)
        with patch(
            "trillim.components.tts._worker.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ), patch(
            "trillim.components.tts._worker._collect_worker_output",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ), patch(
            "trillim.components.tts._worker._stop_process",
            new=AsyncMock(),
        ) as stop_process:
            with self.assertRaises(asyncio.CancelledError):
                await synthesize_segment(
                    "tiny prompt",
                    voice_kind="predefined",
                    voice_reference="alba",
                )
        stop_process.assert_awaited_once_with(process)

        process = SimpleNamespace(returncode=None)
        with patch(
            "trillim.components.tts._worker.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ), patch(
            "trillim.components.tts._worker._collect_worker_output",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ), patch(
            "trillim.components.tts._worker._stop_process",
            new=AsyncMock(),
        ) as stop_process:
            with self.assertRaises(asyncio.CancelledError):
                await build_voice_state(self.root / "voice.txt")
        stop_process.assert_awaited_once_with(process)

        class _CollectedProcess:
            def __init__(self) -> None:
                self.stdout = asyncio.StreamReader()
                self.stderr = asyncio.StreamReader()
                self.returncode = None
                self.stdout.feed_data(b"pcm")
                self.stdout.feed_eof()
                self.stderr.feed_data(b"stderr")
                self.stderr.feed_eof()

            async def wait(self) -> int:
                self.returncode = 0
                return 0

        stdout, stderr = await _collect_worker_output(
            _CollectedProcess(),
            stdout_limit=8,
            timeout=None,
        )
        self.assertEqual(stdout, b"pcm")
        self.assertEqual(stderr, b"stderr")

        class _HangingProcess:
            def __init__(self) -> None:
                self.returncode = None
                self.terminate_calls = 0
                self.kill_calls = 0
                self.wait_calls = 0

            def terminate(self) -> None:
                self.terminate_calls += 1

            def kill(self) -> None:
                self.kill_calls += 1
                self.returncode = -9

            async def wait(self) -> int:
                self.wait_calls += 1
                return 0 if self.returncode is None else self.returncode

        async def fake_wait_for(awaitable, timeout):
            del timeout
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError

        hanging = _HangingProcess()
        with patch("trillim.components.tts._worker.asyncio.wait_for", side_effect=fake_wait_for):
            await _stop_process(hanging)
        self.assertEqual(hanging.terminate_calls, 1)
        self.assertEqual(hanging.kill_calls, 1)
        self.assertGreaterEqual(hanging.wait_calls, 1)
        self.assertEqual(_error_message(b" \n"), "TTS worker failed")

    async def test_persistent_worker_private_helpers_cover_stderr_waits_and_short_reads(self):
        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )

        process = SimpleNamespace(returncode=0, stdin=None)
        stderr_task = asyncio.create_task(asyncio.sleep(0))
        worker._process = process
        worker._stderr_task = stderr_task
        with patch("trillim.components.tts._worker._stop_process", new=AsyncMock()) as stop_process:
            await worker.close()
        stop_process.assert_awaited_once_with(process)

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        wait_process = SimpleNamespace(returncode=None, wait=AsyncMock(return_value=1))
        worker._stderr_task = asyncio.create_task(asyncio.sleep(0))
        await worker._await_process_error(wait_process)
        wait_process.wait.assert_awaited_once()

        class _OversizedReader:
            async def readexactly(self, size: int) -> bytes:
                if size == _RESPONSE_HEADER.size:
                    return _RESPONSE_HEADER.pack(b"A", 2)
                raise AssertionError("payload should not be read after oversize header")

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        worker._stderr_task = None
        with patch("trillim.components.tts._worker.MAX_PCM_CHUNK_BYTES", 1):
            with self.assertRaisesRegex(Exception, "oversized stdout"):
                await worker._read_response(
                    SimpleNamespace(stdout=_OversizedReader(), returncode=None)
                )

    def test_run_session_worker_rejects_short_payload_reads(self):
        class _FakeModel:
            def generate_audio(self, model_state, text_to_generate, max_tokens=20):
                del model_state, text_to_generate, max_tokens
                return [0.0]

        fake_pocket_tts = SimpleNamespace(
            TTSModel=SimpleNamespace(load_model=lambda: _FakeModel())
        )
        stdin_buffer = io.BytesIO(_REQUEST_HEADER.pack(4) + b"ab")
        stdout_buffer = io.BytesIO()

        with patch.dict(sys.modules, {"pocket_tts": fake_pocket_tts}), patch(
            "trillim.components.tts._worker._load_worker_state",
            return_value={"prompt": "voice"},
        ), patch(
            "trillim.components.tts._worker.sys.stdin",
            SimpleNamespace(buffer=stdin_buffer),
        ), patch(
            "trillim.components.tts._worker.sys.stdout",
            SimpleNamespace(buffer=stdout_buffer),
        ):
            with self.assertRaisesRegex(RuntimeError, "malformed stdin input"):
                _run_session_worker(
                    voice_kind="predefined",
                    voice_reference="alba",
                )

    def test_worker_module_main_raises_system_exit_under_dunder_main(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")

        class _FakeModel:
            def get_state_for_audio_prompt(self, audio_conditioning):
                return {"prompt": Path(audio_conditioning).read_text(encoding="latin-1")}

        fake_pocket_tts = SimpleNamespace(
            TTSModel=SimpleNamespace(load_model=lambda: _FakeModel())
        )
        argv = [
            "trillim.components.tts._worker",
            "voice-state",
            "--audio-path",
            str(audio_path),
        ]
        stdout = SimpleNamespace(buffer=io.BytesIO())

        with patch.dict(sys.modules, {"pocket_tts": fake_pocket_tts}), patch.object(
            sys,
            "argv",
            argv,
        ), patch(
            "sys.stdout",
            stdout,
        ):
            with self.assertRaises(SystemExit) as exc:
                runpy.run_path(str(Path(tts_worker_module.__file__)), run_name="__main__")

        self.assertEqual(exc.exception.code, 0)

    def test_worker_entrypoints_cover_direct_session_and_cli_branches(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")

        class _FakeModel:
            def get_state_for_audio_prompt(self, audio_conditioning):
                text = str(audio_conditioning)
                if text in {"alba", "marius"}:
                    return {"prompt": text}
                return {"prompt": Path(text).read_text(encoding="latin-1")}

            def generate_audio(self, model_state, text_to_generate, max_tokens=20):
                del model_state, text_to_generate, max_tokens
                return [0.0, 0.25, -0.25]

        fake_pocket_tts = SimpleNamespace(
            TTSModel=SimpleNamespace(load_model=lambda: _FakeModel())
        )

        synth_stdout = SimpleNamespace(buffer=io.BytesIO())
        with patch.dict(sys.modules, {"pocket_tts": fake_pocket_tts}), patch(
            "trillim.components.tts._worker.sys.stdout",
            synth_stdout,
        ):
            self.assertEqual(
                main(
                    [
                        "synthesize",
                        "--text",
                        "tiny prompt",
                        "--voice-kind",
                        "predefined",
                        "--voice-reference",
                        "alba",
                    ]
                ),
                0,
            )
        self.assertTrue(synth_stdout.buffer.getvalue())

        state_stdout = SimpleNamespace(buffer=io.BytesIO())
        with patch.dict(sys.modules, {"pocket_tts": fake_pocket_tts}), patch(
            "trillim.components.tts._worker.sys.stdout",
            state_stdout,
        ):
            self.assertEqual(
                main(["voice-state", "--audio-path", str(audio_path)]),
                0,
            )
        self.assertIn("prompt", load_safe_voice_state_bytes(state_stdout.buffer.getvalue()))

        request = _REQUEST_HEADER.pack(len(b"hello")) + b"hello"
        session_stdin = SimpleNamespace(buffer=io.BytesIO(request))
        session_stdout = SimpleNamespace(buffer=io.BytesIO())
        with patch.dict(sys.modules, {"pocket_tts": fake_pocket_tts}), patch(
            "trillim.components.tts._worker.sys.stdin",
            session_stdin,
        ), patch(
            "trillim.components.tts._worker.sys.stdout",
            session_stdout,
        ):
            self.assertEqual(
                _run_session_worker(
                    voice_kind="predefined",
                    voice_reference="alba",
                ),
                0,
            )
        payload = session_stdout.buffer.getvalue()
        kind, size = _RESPONSE_HEADER.unpack(payload[: _RESPONSE_HEADER.size])
        self.assertEqual(kind, b"A")
        self.assertEqual(len(payload) - _RESPONSE_HEADER.size, size)

        bad_stdin = SimpleNamespace(buffer=io.BytesIO(b"\x00\x00"))
        bad_stdout = SimpleNamespace(buffer=io.BytesIO())
        bad_stderr = io.StringIO()
        with patch.dict(sys.modules, {"pocket_tts": fake_pocket_tts}), patch(
            "trillim.components.tts._worker.sys.stdin",
            bad_stdin,
        ), patch(
            "trillim.components.tts._worker.sys.stdout",
            bad_stdout,
        ), patch(
            "trillim.components.tts._worker.sys.stderr",
            bad_stderr,
        ):
            self.assertEqual(
                main(
                    [
                        "session",
                        "--voice-kind",
                        "predefined",
                        "--voice-reference",
                        "alba",
                    ]
                ),
                1,
            )
        self.assertIn("malformed stdin", bad_stderr.getvalue())
