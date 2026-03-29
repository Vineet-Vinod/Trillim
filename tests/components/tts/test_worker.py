"""Tests for the TTS worker subprocess helpers."""

from __future__ import annotations

import asyncio
import io
import os
import runpy
import struct
import sys
import tempfile
import types
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
    _bind_stateful_module_names,
    _collect_worker_output,
    _error_message,
    _load_worker_state,
    _run_session_worker,
    _stop_process,
    _worker_command,
    build_voice_state,
    create_session_worker,
    is_voice_cloning_auth_error,
    main,
    synthesize_segment,
)
from tests.components.tts.support import prepend_pythonpath, write_fake_pocket_tts_package


class _UnsafeState:
    pass


class _FakeStatefulModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._module_absolute_name = None

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_state(self, model_state: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return model_state[self._module_absolute_name]


def _fake_init_states(
    model: torch.nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, _FakeStatefulModule):
            continue
        module._module_absolute_name = module_name
        result[module_name] = module.init_state(batch_size, sequence_length=sequence_length)
    return result


_FAKE_STATEFUL_MODULE = types.ModuleType("pocket_tts.modules.stateful_module")
_FAKE_STATEFUL_MODULE.StatefulModule = _FakeStatefulModule
_FAKE_STATEFUL_MODULE.init_states = _fake_init_states


class _FakeStreamingAttention(_FakeStatefulModule):
    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        return {
            "cache": torch.zeros((2, batch_size, sequence_length, 1, 1)),
            "current_end": torch.zeros((0,)),
        }


class _FakeFlowLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeStreamingAttention()


class _FakeFlowLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.layers = torch.nn.ModuleList([_FakeFlowLayer()])


class _FakeStatefulPocketTTSModel:
    def __init__(self) -> None:
        self.flow_lm = _FakeFlowLM()

    def get_state_for_audio_prompt(self, audio_conditioning):
        del audio_conditioning
        return _fake_init_states(
            self.flow_lm,
            batch_size=1,
            sequence_length=2,
        )

    def generate_audio(self, model_state, text_to_generate, max_tokens=20):
        del text_to_generate, max_tokens
        for module in self.flow_lm.modules():
            if isinstance(module, _FakeStatefulModule):
                module.get_state(model_state)
        return [0.0, 0.25, -0.25]


def _stateful_voice_state(model: _FakeStatefulPocketTTSModel) -> dict[str, dict[str, torch.Tensor]]:
    return _fake_init_states(
        model.flow_lm,
        batch_size=1,
        sequence_length=2,
    )


def _fake_stateful_pocket_tts_modules(model: _FakeStatefulPocketTTSModel) -> dict[str, object]:
    package = types.ModuleType("pocket_tts")
    package.__path__ = []
    package.TTSModel = SimpleNamespace(load_model=lambda: model)
    modules_package = types.ModuleType("pocket_tts.modules")
    modules_package.__path__ = []
    modules_package.stateful_module = _FAKE_STATEFUL_MODULE
    package.modules = modules_package
    return {
        "pocket_tts": package,
        "pocket_tts.modules": modules_package,
        "pocket_tts.modules.stateful_module": _FAKE_STATEFUL_MODULE,
    }


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
        model = _FakeStatefulPocketTTSModel()
        fake_modules = _fake_stateful_pocket_tts_modules(model)
        good_path = self.root / "voice.state"
        torch.save(_stateful_voice_state(model), good_path)
        with patch.dict(sys.modules, fake_modules):
            loaded_state = _load_worker_state(
                model,
                voice_kind="state_file",
                voice_reference=str(good_path),
            )
        self.assertEqual(list(loaded_state), ["transformer.layers.0.self_attn"])
        self.assertEqual(
            sorted(loaded_state["transformer.layers.0.self_attn"]),
            ["cache", "current_end"],
        )

        bad_path = self.root / "bad.state"
        torch.save({"bad": _UnsafeState()}, bad_path)
        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(Exception, "voice state is malformed"):
                _load_worker_state(
                    model,
                    voice_kind="state_file",
                    voice_reference=str(bad_path),
                )

        with self.assertRaisesRegex(ValueError, "unsupported voice kind"):
            _load_worker_state(
                SimpleNamespace(),
                voice_kind="unknown",
                voice_reference="ignored",
            )

    def test_load_worker_state_binds_names_and_rejects_incompatible_saved_state_early(self):
        model = _FakeStatefulPocketTTSModel()
        fake_modules = _fake_stateful_pocket_tts_modules(model)
        stateful_modules = [
            module
            for module in model.flow_lm.modules()
            if isinstance(module, _FakeStatefulModule)
        ]
        self.assertEqual(
            [module._module_absolute_name for module in stateful_modules],
            [None],
        )

        good_path = self.root / "good.state"
        torch.save(_stateful_voice_state(model), good_path)
        with patch.dict(sys.modules, fake_modules):
            _load_worker_state(
                model,
                voice_kind="state_file",
                voice_reference=str(good_path),
            )
        self.assertEqual(
            [module._module_absolute_name for module in stateful_modules],
            ["transformer.layers.0.self_attn"],
        )

        wrong_module_path = self.root / "wrong-module.state"
        wrong_module_state = {
            "wrong.module": _stateful_voice_state(model)["transformer.layers.0.self_attn"]
        }
        torch.save(wrong_module_state, wrong_module_path)
        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "missing module state"):
                _load_worker_state(
                    model,
                    voice_kind="state_file",
                    voice_reference=str(wrong_module_path),
                )

        missing_key_path = self.root / "missing-key.state"
        missing_key_state = {
            "transformer.layers.0.self_attn": {"cache": torch.zeros((2, 1, 2, 1, 1))}
        }
        torch.save(missing_key_state, missing_key_path)
        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "missing keys 'current_end'"):
                _load_worker_state(
                    model,
                    voice_kind="state_file",
                    voice_reference=str(missing_key_path),
                )

        unexpected_key_path = self.root / "unexpected-key.state"
        unexpected_key_state = {
            "transformer.layers.0.self_attn": {
                "cache": torch.zeros((2, 1, 2, 1, 1)),
                "current_end": torch.zeros((0,)),
                "extra": torch.ones((1,)),
            }
        }
        torch.save(unexpected_key_state, unexpected_key_path)
        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "unexpected keys 'extra'"):
                _load_worker_state(
                    model,
                    voice_kind="state_file",
                    voice_reference=str(unexpected_key_path),
                )

        malformed_module_state_path = self.root / "malformed-module.state"
        malformed_module_state = {
            "transformer.layers.0.self_attn": torch.zeros((1,))
        }
        torch.save(malformed_module_state, malformed_module_state_path)
        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "is malformed"):
                _load_worker_state(
                    model,
                    voice_kind="state_file",
                    voice_reference=str(malformed_module_state_path),
                )

        unexpected_module_path = self.root / "unexpected-module.state"
        unexpected_module_state = {
            "transformer.layers.0.self_attn": _stateful_voice_state(model)["transformer.layers.0.self_attn"],
            "extra0": {},
            "extra1": {},
            "extra2": {},
            "extra3": {},
            "extra4": {},
        }
        torch.save(unexpected_module_state, unexpected_module_path)
        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(
                RuntimeError,
                r"unexpected module state for 'extra0', 'extra1', 'extra2', 'extra3', \.\.\.",
            ):
                _load_worker_state(
                    model,
                    voice_kind="state_file",
                    voice_reference=str(unexpected_module_path),
                )

    def test_bind_stateful_module_names_rejects_missing_or_malformed_models(self):
        base_model = _FakeStatefulPocketTTSModel()
        fake_modules = _fake_stateful_pocket_tts_modules(base_model)

        class _NoStatefulFlowLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

        class _NoStatefulModel:
            def __init__(self) -> None:
                self.flow_lm = _NoStatefulFlowLM()

        class _MalformedStatefulModule(_FakeStatefulModule):
            def init_state(self, batch_size: int, sequence_length: int):
                del batch_size, sequence_length
                return ["bad-state"]

        class _MalformedFlowLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.transformer.layers[0].self_attn = _MalformedStatefulModule()

        class _MalformedModel:
            def __init__(self) -> None:
                self.flow_lm = _MalformedFlowLM()

        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(RuntimeError, "model is missing flow_lm"):
                _bind_stateful_module_names(SimpleNamespace())
            with self.assertRaisesRegex(RuntimeError, "model has no stateful modules"):
                _bind_stateful_module_names(_NoStatefulModel())
            with self.assertRaisesRegex(RuntimeError, "produced malformed state"):
                _bind_stateful_module_names(_MalformedModel())

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

    def test_run_session_worker_streams_saved_state_voices_like_built_ins(self):
        model = _FakeStatefulPocketTTSModel()
        state_path = self.root / "voice.state"
        torch.save(_stateful_voice_state(model), state_path)
        fake_modules = _fake_stateful_pocket_tts_modules(model)

        def run_once(*, voice_kind: str, voice_reference: str) -> bytes:
            request = _REQUEST_HEADER.pack(len(b"hello")) + b"hello"
            stdin_buffer = io.BytesIO(request)
            stdout_buffer = io.BytesIO()
            with patch.dict(sys.modules, fake_modules), patch(
                "trillim.components.tts._worker.sys.stdin",
                SimpleNamespace(buffer=stdin_buffer),
            ), patch(
                "trillim.components.tts._worker.sys.stdout",
                SimpleNamespace(buffer=stdout_buffer),
            ):
                self.assertEqual(
                    _run_session_worker(
                        voice_kind=voice_kind,
                        voice_reference=voice_reference,
                    ),
                    0,
                )
            return stdout_buffer.getvalue()

        built_in_payload = run_once(
            voice_kind="predefined",
            voice_reference="alba",
        )
        saved_state_payload = run_once(
            voice_kind="state_file",
            voice_reference=str(state_path),
        )
        self.assertEqual(saved_state_payload, built_in_payload)

    async def test_worker_command_bootstrap_avoids_runpy_warning(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")
        with patch.dict(os.environ, prepend_pythonpath(self.root)):
            process = await asyncio.create_subprocess_exec(
                *_worker_command("voice-state", audio_path=str(audio_path)),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        self.assertEqual(process.returncode, 0)
        self.assertTrue(stdout)
        self.assertEqual(stderr, b"")

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
