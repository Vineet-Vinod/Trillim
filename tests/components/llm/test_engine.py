"""Tests for the LLM inference engine protocol."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import trillim.components.llm._engine as engine_module
from trillim.components.llm._config import InitConfig, SamplingDefaults
from trillim.components.llm._engine import (
    EngineCrashedError,
    EngineProgressTimeoutError,
    EngineProtocolError,
    InferenceEngine,
    _PromptCache,
    _bundled_binary_path,
    _first_protocol_line,
    _read_stderr,
)
from tests.components.llm.support import FakeTokenizer, make_runtime_model


class _FakeWriter:
    def __init__(self, *, drain_error: Exception | None = None):
        self.drain_error = drain_error
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        if self.drain_error is not None:
            raise self.drain_error


class _FakeReader:
    def __init__(self, lines=None, *, read_data: bytes = b""):
        self.lines = list(lines or [])
        self.read_data = read_data

    async def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        return b""

    async def read(self) -> bytes:
        return self.read_data


class _FakeProcess:
    def __init__(self, *, lines=None, stderr_data: bytes = b"", drain_error: Exception | None = None):
        self.stdin = _FakeWriter(drain_error=drain_error)
        self.stdout = _FakeReader(lines)
        self.stderr = _FakeReader(read_data=stderr_data)
        self.returncode = None
        self.killed = False
        self.wait_calls = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class EngineTests(unittest.IsolatedAsyncioTestCase):
    def _expected_binary_path(self, *, os_name: str | None = None, suffix: str | None = None) -> str:
        resolved_os_name = engine_module.os.name if os_name is None else os_name
        resolved_suffix = ".exe" if resolved_os_name == "nt" else ""
        if suffix is not None:
            resolved_suffix = suffix
        return str(
            Path(engine_module.__file__).resolve().parents[2]
            / "_bin"
            / f"trillim-inference{resolved_suffix}"
        )

    def _make_engine(self) -> InferenceEngine:
        return InferenceEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            SamplingDefaults(),
            progress_timeout=5.0,
        )

    def test_inference_engine_does_not_expose_binary_path_override(self):
        self.assertNotIn("binary_path", inspect.signature(InferenceEngine).parameters)

    def test_inference_engine_raises_when_bundled_binary_is_missing(self):
        with patch.object(engine_module.Path, "is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Missing bundled LLM inference binary"):
                self._make_engine()

    def test_bundled_binary_path_uses_windows_suffix_and_fallback(self):
        exe_path = Path(self._expected_binary_path(os_name="nt", suffix=".exe"))
        fallback_path = Path(self._expected_binary_path(os_name="nt", suffix=""))

        with patch.object(engine_module, "os", SimpleNamespace(name="nt")), patch.object(
            engine_module.Path,
            "is_file",
            autospec=True,
            side_effect=lambda path: path == exe_path,
        ):
            self.assertEqual(_bundled_binary_path(), str(exe_path))

        with patch.object(engine_module, "os", SimpleNamespace(name="nt")), patch.object(
            engine_module.Path,
            "is_file",
            autospec=True,
            side_effect=lambda path: path == fallback_path,
        ):
            self.assertEqual(_bundled_binary_path(), str(fallback_path))

    def test_bundled_binary_path_covers_windows_missing_fallback_and_non_windows_recheck(self):
        with patch.object(engine_module, "os", SimpleNamespace(name="nt")), patch.object(
            engine_module.Path,
            "is_file",
            autospec=True,
            return_value=False,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "Missing bundled LLM inference binary"):
                _bundled_binary_path()

        bundled_path = Path(self._expected_binary_path(os_name="posix"))
        with patch.object(engine_module, "os", SimpleNamespace(name="posix")), patch.object(
            engine_module.Path,
            "is_file",
            autospec=True,
            return_value=True,
        ):
            self.assertEqual(_bundled_binary_path(), str(bundled_path))

    async def test_start_launches_process_and_writes_init_block(self):
        engine = self._make_engine()
        process = _FakeProcess()

        with patch(
            "trillim.components.llm._engine.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ) as mock_exec:
            await engine.start()

        mock_exec.assert_awaited_once()
        self.assertEqual(mock_exec.await_args.args[0], self._expected_binary_path())
        self.assertTrue(process.stdin.writes[0].startswith(b"17\n"))

    async def test_start_writes_adapter_init_fields_when_configured(self):
        engine = InferenceEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            SamplingDefaults(),
            init_config=InitConfig(
                model_dir=Path("/tmp/model"),
                num_threads=4,
                lora_dir=Path("/tmp/adapter"),
                lora_quant="q4_0",
                unembed_quant="q8_0",
            ),
            progress_timeout=5.0,
        )
        process = _FakeProcess()

        with patch(
            "trillim.components.llm._engine.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ):
            await engine.start()

        block = process.stdin.writes[0].decode("utf-8")
        self.assertIn("num_threads=4\n", block)
        self.assertIn("lora_dir=/tmp/adapter\n", block)
        self.assertIn("lora_quant=q4_0\n", block)
        self.assertIn("unembed_quant=q8_0\n", block)

    async def test_start_uses_only_the_first_line_of_string_init_fields(self):
        engine = InferenceEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            SamplingDefaults(),
            init_config=InitConfig(
                model_dir=Path("/tmp/model"),
                num_threads=4,
                lora_dir=Path("/tmp/adapter\nnum_threads=999"),
                lora_quant="q4_0\nrope_theta=1",
                unembed_quant="q8_0\nactivation=0",
            ),
            progress_timeout=5.0,
        )
        process = _FakeProcess()

        with patch(
            "trillim.components.llm._engine.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ):
            await engine.start()

        block = process.stdin.writes[0].decode("utf-8")
        self.assertIn("num_threads=4\n", block)
        self.assertIn("lora_dir=/tmp/adapter\n", block)
        self.assertIn("lora_quant=q4_0\n", block)
        self.assertIn("unembed_quant=q8_0\n", block)
        self.assertNotIn("num_threads=999\n", block)
        self.assertNotIn("rope_theta=1\n", block)
        self.assertNotIn("activation=0\n", block)

    async def test_stop_writes_exit_block(self):
        engine = self._make_engine()
        process = _FakeProcess()
        engine.process = process

        await engine.stop()

        self.assertEqual(process.stdin.writes, [b"0\n"])
        self.assertEqual(process.wait_calls, 1)

    async def test_generate_reuses_cached_prefix_and_updates_cache(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(lines=[b"65\n", b"0\n", b"4\n"])
        engine._prompt_cache._token_ids = (1, 2)

        tokens = [token async for token in engine.generate([1, 2, 99], max_tokens=8)]

        self.assertEqual(tokens, [65])
        request_block = engine.process.stdin.writes[0].decode("utf-8")
        self.assertIn("reset=0\n", request_block)
        self.assertIn("tokens=99\n", request_block)
        self.assertEqual(engine.last_cache_hit, 2)
        self.assertEqual(engine.last_prompt_tokens, 3)
        self.assertEqual(engine.last_completion_tokens, 1)
        self.assertEqual(engine.cached_token_ids, [1, 2, 99, 65])

    async def test_generate_resets_cache_when_prompt_prefix_diverges(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(lines=[b"65\n", b"0\n", b"3\n"])
        engine._prompt_cache._token_ids = (1, 2, 3)

        tokens = [token async for token in engine.generate([1, 9], max_tokens=8)]

        self.assertEqual(tokens, [65])
        request_block = engine.process.stdin.writes[0].decode("utf-8")
        self.assertIn("reset=1\n", request_block)
        self.assertIn("tokens=1,9\n", request_block)
        self.assertEqual(engine.last_cache_hit, 0)
        self.assertEqual(engine.cached_token_ids, [1, 9, 65])

    async def test_generate_usage_follows_authoritative_kv_position(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(lines=[b"65\n", b"66\n", b"0\n", b"4\n"])

        tokens = [token async for token in engine.generate([1, 2, 3], max_tokens=8)]

        self.assertEqual(tokens, [65, 66])
        self.assertEqual(engine.last_prompt_tokens, 3)
        self.assertEqual(engine.last_completion_tokens, 1)
        self.assertEqual(engine.cached_token_count, 4)

    async def test_generate_raises_protocol_errors_for_invalid_ints(self):
        engine = self._make_engine()
        process = _FakeProcess(lines=[b"oops\n"])
        engine.process = process

        with self.assertRaisesRegex(EngineProtocolError, "expected int token_id"):
            [token async for token in engine.generate([1], max_tokens=8)]

        self.assertTrue(process.killed)

    async def test_generate_times_out_when_progress_stalls(self):
        engine = self._make_engine()
        process = _FakeProcess(lines=[b"65\n"])
        engine.process = process

        async def fake_wait_for(awaitable, timeout):
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError

        with patch("trillim.components.llm._engine.asyncio.wait_for", side_effect=fake_wait_for):
            with self.assertRaisesRegex(EngineProgressTimeoutError, "no write progress"):
                [token async for token in engine.generate([1], max_tokens=8)]

    async def test_generate_surfaces_engine_crashes(self):
        engine = self._make_engine()
        process = _FakeProcess(lines=[b""], stderr_data=b"boom")
        engine.process = process

        with self.assertRaisesRegex(EngineCrashedError, "boom"):
            [token async for token in engine.generate([1], max_tokens=8)]

    def test_prompt_cache_commit_rejects_invalid_kv_positions(self):
        cache = _PromptCache()
        plan = cache.plan([1, 2])

        with self.assertRaisesRegex(EngineProtocolError, "Invalid kv_position -1"):
            cache.commit(plan, [65], -1)

    async def test_start_stop_and_generate_helper_paths(self):
        engine = self._make_engine()
        running = _FakeProcess()
        engine.process = running
        with patch(
            "trillim.components.llm._engine.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=AssertionError("should not launch")),
        ):
            await engine.start()

        engine = self._make_engine()
        failed_process = _FakeProcess()
        with patch.object(
            engine,
            "_write_block",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch(
            "trillim.components.llm._engine.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=failed_process),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await engine.start()
        self.assertTrue(failed_process.killed)
        self.assertIsNone(engine.process)

        engine = self._make_engine()
        await engine.stop()
        dead_process = _FakeProcess()
        dead_process.returncode = 0
        engine.process = dead_process
        await engine.stop()

        broken_process = _FakeProcess(drain_error=BrokenPipeError("boom"))
        engine.process = broken_process
        await engine.stop()
        self.assertTrue(broken_process.killed)

        cancelled_process = _FakeProcess()
        engine.process = cancelled_process
        with patch.object(
            engine,
            "_write_block",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with self.assertRaises(asyncio.CancelledError):
                [token async for token in engine.generate([1], max_tokens=8)]
        self.assertTrue(cancelled_process.killed)

        oserror_process = _FakeProcess()
        engine.process = oserror_process
        with patch.object(
            engine,
            "_write_block",
            new=AsyncMock(side_effect=OSError("boom")),
        ):
            with self.assertRaisesRegex(EngineCrashedError, "Inference engine crashed"):
                [token async for token in engine.generate([1], max_tokens=8)]
        self.assertTrue(oserror_process.killed)

    async def test_engine_low_level_helpers_cover_missing_pipes_and_fallbacks(self):
        engine = self._make_engine()
        with self.assertRaisesRegex(EngineCrashedError, "not running"):
            engine._require_running()

        process = _FakeProcess()
        process.stdin = None
        engine.process = process
        with self.assertRaisesRegex(EngineCrashedError, "stdin is unavailable"):
            await engine._write_block("demo")

        process = _FakeProcess()
        process.stdout = None
        engine.process = process
        with self.assertRaisesRegex(EngineCrashedError, "stdout is unavailable"):
            await engine._readline("token_id")

        process = _FakeProcess(lines=[b""], stderr_data=b"")
        engine.process = process
        with self.assertRaisesRegex(EngineCrashedError, "Inference engine crashed$"):
            await engine._readline("token_id")

        engine.process = None
        await engine._kill_process()

        async def fake_wait_for(awaitable, timeout):
            del timeout
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError

        process = _FakeProcess(lines=[b"65\n"])
        engine.process = process
        with patch("trillim.components.llm._engine.asyncio.wait_for", side_effect=fake_wait_for):
            with self.assertRaisesRegex(EngineProgressTimeoutError, "no token_id progress"):
                await engine._readline("token_id")

        idle_process = _FakeProcess()
        idle_process.returncode = 0
        engine.process = idle_process
        await engine._kill_process()
        self.assertIsNone(engine.process)

        oserror_process = _FakeProcess()

        def kill_with_error() -> None:
            raise OSError("gone")

        oserror_process.kill = kill_with_error
        engine.process = oserror_process
        await engine._kill_process()
        self.assertIsNone(engine.process)

        self.assertEqual(_first_protocol_line("   "), "   ")
        self.assertEqual(_first_protocol_line("first\nsecond"), "first")
        self.assertEqual(await _read_stderr(SimpleNamespace(stderr=None)), "")

        class _BadReader:
            async def read(self) -> bytes:
                raise RuntimeError("boom")

        self.assertEqual(await _read_stderr(SimpleNamespace(stderr=_BadReader())), "")

    async def test_generate_keeps_live_process_reference_when_kill_is_noop(self):
        engine = self._make_engine()
        process = _FakeProcess(lines=[b"65\n"])
        engine.process = process

        async def noop_kill() -> None:
            return None

        with patch.object(
            engine,
            "_write_block",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ), patch.object(engine, "_kill_process", new=AsyncMock(side_effect=noop_kill)):
            with self.assertRaises(asyncio.CancelledError):
                [token async for token in engine.generate([1], max_tokens=8)]

        self.assertIs(engine.process, process)
