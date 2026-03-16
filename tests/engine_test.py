# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the async inference engine protocol handling."""

import asyncio
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from trillim._prompt_cache import PromptSnapshot
from trillim.engine import _ENGINE_TIMEOUT, InferenceEngine


class _TokenizerStub:
    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(ch) for ch in text]


class _FakeWriter:
    def __init__(
        self,
        *,
        write_error: Exception | None = None,
        drain_error: Exception | None = None,
    ):
        self.write_error = write_error
        self.drain_error = drain_error
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.writes.append(data)

    async def drain(self) -> None:
        if self.drain_error is not None:
            raise self.drain_error


class _FakeReader:
    def __init__(
        self,
        lines: list[bytes] | None = None,
        *,
        read_data: bytes = b"",
        read_error: Exception | None = None,
    ):
        self._lines = list(lines or [])
        self._read_data = read_data
        self._read_error = read_error

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""

    async def read(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._read_data


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[bytes] | None = None,
        stderr_data: bytes = b"",
        stderr_error: Exception | None = None,
        write_error: Exception | None = None,
        drain_error: Exception | None = None,
        returncode: int | None = None,
    ):
        self.stdin = _FakeWriter(write_error=write_error, drain_error=drain_error)
        self.stdout = _FakeReader(stdout_lines)
        self.stderr = _FakeReader(read_data=stderr_data, read_error=stderr_error)
        self.returncode = returncode
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


class _KillOSErrorProcess(_FakeProcess):
    def kill(self) -> None:
        raise OSError("kill failed")


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class InferenceEngineTests(unittest.IsolatedAsyncioTestCase):
    def _make_engine(self, **kwargs) -> InferenceEngine:
        return InferenceEngine(
            model_dir="models/fake",
            tokenizer=_TokenizerStub(),
            stop_tokens={0},
            default_params={
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "rep_penalty_lookback": 32,
            },
            arch_config=SimpleNamespace(),
            **kwargs,
        )

    @staticmethod
    def _seed_cache(
        engine: InferenceEngine,
        token_ids: list[int],
        *,
        last_cache_hit: int = 0,
    ) -> None:
        engine._prompt_cache.restore(
            PromptSnapshot.create(token_ids),
            last_cache_hit=last_cache_hit,
        )

    async def _collect(self, engine: InferenceEngine, **kwargs) -> list[int]:
        tokens: list[int] = []
        async for token_id in engine.generate(**kwargs):
            tokens.append(token_id)
        return tokens

    @staticmethod
    def _close_awaitable(awaitable) -> None:
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()

    async def test_start_launches_process_and_writes_init_block(self):
        engine = self._make_engine(num_threads=4, lora_quant="q4", unembed_quant="q8")
        proc = _FakeProcess()

        utils_module = _module(
            "trillim.utils",
            _build_init_config=lambda arch_config, adapter_dir=None, **options: (
                f"init:{adapter_dir}:{options['num_threads']}:{options['lora_quant']}:{options['unembed_quant']}"
            ),
            load_engine_options=lambda **kwargs: kwargs,
        )
        bin_path_module = _module("trillim._bin_path", inference_bin=lambda: "/fake/infer")

        with (
            patch.dict(sys.modules, {
                "trillim.utils": utils_module,
                "trillim._bin_path": bin_path_module,
            }),
            patch("trillim.engine.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as mock_exec,
        ):
            await engine.start()

        self.assertIs(engine.process, proc)
        mock_exec.assert_awaited_once_with(
            "/fake/infer",
            "models/fake",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.assertEqual(proc.stdin.writes, [b"init:None:4:q4:q8"])

    async def test_start_rejects_unquantized_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            engine = self._make_engine(adapter_dir=str(adapter_dir))
            utils_module = _module(
                "trillim.utils",
                _build_init_config=lambda *args, **kwargs: "unused",
                load_engine_options=lambda **kwargs: kwargs,
            )

            with (
                patch.dict(sys.modules, {"trillim.utils": utils_module}),
                self.assertRaisesRegex(RuntimeError, "This adapter has not been quantized"),
            ):
                await engine.start()

    async def test_start_rejects_adapter_without_lora_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "trillim_config.json").write_text("{}", encoding="utf-8")
            engine = self._make_engine(adapter_dir=str(adapter_dir))
            utils_module = _module(
                "trillim.utils",
                _build_init_config=lambda *args, **kwargs: "unused",
                load_engine_options=lambda **kwargs: kwargs,
            )

            with (
                patch.dict(sys.modules, {"trillim.utils": utils_module}),
                self.assertRaisesRegex(RuntimeError, "--lora set but .*qmodel\\.lora not found"),
            ):
                await engine.start()

    async def test_start_validates_adapter_compatibility_when_files_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "trillim_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "qmodel.lora").write_bytes(b"lora")
            engine = self._make_engine(adapter_dir=str(adapter_dir))
            proc = _FakeProcess()
            validate_calls: list[tuple[str, str]] = []

            utils_module = _module(
                "trillim.utils",
                _build_init_config=lambda arch_config, adapter_dir=None, **options: "adapter-init",
                load_engine_options=lambda **kwargs: kwargs,
            )
            bin_path_module = _module("trillim._bin_path", inference_bin=lambda: "/fake/infer")
            model_store_module = _module(
                "trillim.model_store",
                validate_adapter_model_compat=lambda adapter, model: validate_calls.append((adapter, model)),
            )

            with (
                patch.dict(sys.modules, {
                    "trillim.utils": utils_module,
                    "trillim._bin_path": bin_path_module,
                    "trillim.model_store": model_store_module,
                }),
                patch("trillim.engine.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
            ):
                await engine.start()

        self.assertEqual(validate_calls, [(str(adapter_dir), "models/fake")])
        self.assertEqual(proc.stdin.writes, [b"adapter-init"])

    async def test_stop_resets_cache_without_process(self):
        engine = self._make_engine()
        self._seed_cache(engine, [1, 2], last_cache_hit=2)

        await engine.stop()

        self.assertEqual(engine.cached_token_ids, [])
        self.assertEqual(engine.last_cache_hit, 0)

    async def test_stop_gracefully_terminates_running_process(self):
        engine = self._make_engine()
        proc = _FakeProcess()
        engine.process = proc

        await engine.stop()

        self.assertEqual(proc.stdin.writes, [b"0\n"])
        self.assertEqual(proc.wait_calls, 1)
        self.assertFalse(proc.killed)

    async def test_stop_kills_process_after_io_failure(self):
        engine = self._make_engine()
        proc = _FakeProcess(drain_error=BrokenPipeError())
        engine.process = proc

        await engine.stop()

        self.assertTrue(proc.killed)
        self.assertEqual(proc.wait_calls, 1)

    async def test_generate_requires_running_process(self):
        engine = self._make_engine()

        with self.assertRaisesRegex(RuntimeError, "Inference process is not running"):
            await self._collect(engine, token_ids=[1])

        engine.process = _FakeProcess(returncode=1)
        with self.assertRaisesRegex(RuntimeError, "Inference process is not running"):
            await self._collect(engine, token_ids=[1])

    async def test_generate_reuses_exact_cached_token_prefix(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(stdout_lines=[b"65\n", b"0\n", b"4\n"])
        self._seed_cache(engine, [1, 2])
        request_calls: list[tuple[list[int], int, dict]] = []

        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda delta_tokens, reset_flag, **kwargs: (
                request_calls.append((list(delta_tokens), reset_flag, kwargs)) or "request"
            ),
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            tokens = await self._collect(
                engine,
                token_ids=[1, 2, 99, 100],
            )

        self.assertEqual(tokens, [65])
        self.assertEqual(request_calls, [
            (
                [99, 100],
                0,
                {
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "rep_penalty_lookback": 32,
                    "max_tokens": 0,
                },
            )
        ])
        self.assertEqual(engine.last_cache_hit, 2)
        self.assertEqual(engine.cached_token_ids, [1, 2, 99, 100])

    async def test_generate_resets_when_request_is_shorter_than_cached_prefix(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(stdout_lines=[b"0\n", b"1\n"])
        self._seed_cache(engine, [5, 6])
        request_calls: list[tuple[list[int], int]] = []

        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda delta_tokens, reset_flag, **kwargs: (
                request_calls.append((list(delta_tokens), reset_flag)) or "request"
            ),
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            tokens = await self._collect(
                engine,
                token_ids=[5],
                temperature=0.2,
                top_k=4,
                top_p=0.5,
                repetition_penalty=1.3,
                rep_penalty_lookback=16,
                max_tokens=7,
            )

        self.assertEqual(tokens, [])
        self.assertEqual(request_calls, [([5], 1)])
        self.assertEqual(engine.last_cache_hit, 0)
        self.assertEqual(engine.cached_token_ids, [5])

    async def test_generate_reuses_token_prefix_when_cached_prompt_is_unavailable(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(stdout_lines=[b"0\n", b"3\n"])
        self._seed_cache(engine, [1, 2])
        request_calls: list[tuple[list[int], int]] = []

        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda delta_tokens, reset_flag, **kwargs: (
                request_calls.append((list(delta_tokens), reset_flag)) or "request"
            ),
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            tokens = await self._collect(engine, token_ids=[1, 2, 3])

        self.assertEqual(tokens, [])
        self.assertEqual(request_calls, [([3], 0)])
        self.assertEqual(engine.last_cache_hit, 2)
        self.assertEqual(engine.cached_token_ids, [1, 2, 3])

    async def test_generate_resets_when_token_prefix_only_partially_matches(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(stdout_lines=[b"0\n", b"2\n"])
        self._seed_cache(engine, [1, 9])
        request_calls: list[tuple[list[int], int]] = []

        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda delta_tokens, reset_flag, **kwargs: (
                request_calls.append((list(delta_tokens), reset_flag)) or "request"
            ),
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            await self._collect(engine, token_ids=[1, 2])

        self.assertEqual(request_calls, [([1, 2], 1)])
        self.assertEqual(engine.last_cache_hit, 0)

    async def test_generate_surfaces_broken_pipe_as_engine_crash(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(drain_error=BrokenPipeError(), stderr_data=b"segfault")
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            with self.assertRaisesRegex(RuntimeError, "Inference engine crashed: segfault"):
                await self._collect(engine, token_ids=[1])

        self.assertEqual(engine.cached_token_ids, [])
        self.assertEqual(engine.last_cache_hit, 0)

    async def test_generate_normalizes_sampling_validation_errors_without_killing_process(self):
        engine = self._make_engine()
        proc = _FakeProcess(stdout_lines=[b"65\n", b"0\n", b"2\n"])
        engine.process = proc

        with self.assertRaisesRegex(ValueError, "temperature must be >= 0"):
            await self._collect(engine, token_ids=[1], temperature=-0.1)

        self.assertEqual(proc.stdin.writes, [])
        self.assertFalse(proc.killed)
        self.assertIsNone(proc.returncode)

        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            tokens = await self._collect(engine, token_ids=[1])

        self.assertEqual(tokens, [65])
        self.assertEqual(proc.stdin.writes, [b"request"])

    async def test_generate_raises_on_unexpected_stdout_eof(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(stdout_lines=[b"65\n", b""], stderr_data=b"child exited")
        self._seed_cache(engine, [1], last_cache_hit=1)
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            tokens: list[int] = []
            with self.assertRaisesRegex(RuntimeError, "Inference engine crashed: child exited"):
                async for token_id in engine.generate(token_ids=[1, 2]):
                    tokens.append(token_id)

        self.assertEqual(tokens, [65])
        self.assertEqual(engine.cached_token_ids, [])
        self.assertEqual(engine.last_cache_hit, 0)

    async def test_generate_handles_stderr_read_failures_on_crash(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(stdout_lines=[b""], stderr_error=RuntimeError("stderr failed"))
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            with self.assertRaisesRegex(RuntimeError, "^Inference engine crashed$"):
                await self._collect(engine, token_ids=[1])

    async def test_generate_raises_on_invalid_token_id(self):
        engine = self._make_engine()
        proc = _FakeProcess(stdout_lines=[b"oops\n"])
        engine.process = proc
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            with self.assertRaisesRegex(RuntimeError, "Protocol error: expected int token_id"):
                await self._collect(engine, token_ids=[1])

        self.assertTrue(proc.killed)

    async def test_generate_times_out_while_writing_request(self):
        engine = self._make_engine()
        proc = _FakeProcess()
        engine.process = proc
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )
        real_wait_for = asyncio.wait_for

        async def fake_wait_for(awaitable, timeout):
            self.assertEqual(timeout, _ENGINE_TIMEOUT)
            self._close_awaitable(awaitable)
            raise asyncio.TimeoutError

        with (
            patch.dict(sys.modules, {"trillim.utils": utils_module}),
            patch("trillim.engine.asyncio.wait_for", side_effect=fake_wait_for),
        ):
            with self.assertRaisesRegex(RuntimeError, "Inference engine unresponsive"):
                await self._collect(engine, token_ids=[1])

        self.assertTrue(proc.killed)
        self.assertIs(real_wait_for, asyncio.wait_for)

    async def test_generate_times_out_while_waiting_for_tokens(self):
        engine = self._make_engine()
        proc = _FakeProcess(stdout_lines=[b"65\n", b"0\n", b"3\n"])
        engine.process = proc
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )
        real_wait_for = asyncio.wait_for
        call_count = 0

        async def fake_wait_for(awaitable, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                self._close_awaitable(awaitable)
                raise asyncio.TimeoutError
            return await real_wait_for(awaitable, timeout)

        with (
            patch.dict(sys.modules, {"trillim.utils": utils_module}),
            patch("trillim.engine.asyncio.wait_for", side_effect=fake_wait_for),
        ):
            with self.assertRaisesRegex(RuntimeError, "Inference engine unresponsive"):
                await self._collect(engine, token_ids=[1])

        self.assertTrue(proc.killed)

    async def test_generate_raises_on_invalid_kv_position(self):
        engine = self._make_engine()
        proc = _FakeProcess(stdout_lines=[b"0\n", b"bad\n"])
        engine.process = proc
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            with self.assertRaisesRegex(RuntimeError, "Protocol error: expected int kv_position"):
                await self._collect(engine, token_ids=[1])

        self.assertTrue(proc.killed)

    async def test_generate_kills_process_when_consumer_abandons_request(self):
        engine = self._make_engine()
        proc = _FakeProcess(stdout_lines=[b"65\n"])
        engine.process = proc
        self._seed_cache(engine, [1, 2], last_cache_hit=2)
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            iterator = engine.generate(token_ids=[1, 2, 3])
            self.assertEqual(await anext(iterator), 65)
            await iterator.aclose()

        self.assertTrue(proc.killed)
        self.assertEqual(engine.cached_token_ids, [])
        self.assertEqual(engine.last_cache_hit, 0)

    async def test_generate_times_out_while_waiting_for_kv_position(self):
        engine = self._make_engine()
        proc = _FakeProcess(stdout_lines=[b"0\n", b"1\n"])
        engine.process = proc
        self._seed_cache(engine, [1, 2], last_cache_hit=2)
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )
        real_wait_for = asyncio.wait_for
        call_count = 0

        async def fake_wait_for(awaitable, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                self._close_awaitable(awaitable)
                raise asyncio.TimeoutError
            return await real_wait_for(awaitable, timeout)

        with (
            patch.dict(sys.modules, {"trillim.utils": utils_module}),
            patch("trillim.engine.asyncio.wait_for", side_effect=fake_wait_for),
        ):
            with self.assertRaisesRegex(RuntimeError, "Inference engine unresponsive"):
                await self._collect(engine, token_ids=[1])

        self.assertTrue(proc.killed)
        self.assertEqual(engine.cached_token_ids, [])
        self.assertEqual(engine.last_cache_hit, 0)

    async def test_generate_raises_on_missing_kv_position_after_eos(self):
        engine = self._make_engine()
        engine.process = _FakeProcess(stdout_lines=[b"0\n", b""], stderr_data=b"missing kv")
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            with self.assertRaisesRegex(RuntimeError, "Inference engine crashed: missing kv"):
                await self._collect(engine, token_ids=[1])

    async def test_generate_swallows_oserror_when_cleanup_kill_fails(self):
        engine = self._make_engine()
        engine.process = _KillOSErrorProcess(stdout_lines=[b""], stderr_data=b"crashed")
        utils_module = _module(
            "trillim.utils",
            _build_request_block=lambda *args, **kwargs: "request",
        )

        with patch.dict(sys.modules, {"trillim.utils": utils_module}):
            with self.assertRaisesRegex(RuntimeError, "Inference engine crashed: crashed"):
                await self._collect(engine, token_ids=[1])


if __name__ == "__main__":
    unittest.main()
