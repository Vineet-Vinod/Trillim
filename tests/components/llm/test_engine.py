"""Tests for the LLM inference engine protocol."""

from __future__ import annotations

import asyncio
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from trillim.components.llm._config import SamplingDefaults
from trillim.components.llm._engine import (
    EngineCrashedError,
    EngineProgressTimeoutError,
    EngineProtocolError,
    InferenceEngine,
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
    def _make_engine(self) -> InferenceEngine:
        return InferenceEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            SamplingDefaults(),
            progress_timeout=5.0,
            binary_path="/fake/inference",
        )

    async def test_start_launches_process_and_writes_init_block(self):
        engine = self._make_engine()
        process = _FakeProcess()

        with patch(
            "trillim.components.llm._engine.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ) as mock_exec:
            await engine.start()

        mock_exec.assert_awaited_once()
        self.assertTrue(process.stdin.writes[0].startswith(b"17\n"))

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
        self.assertEqual(engine.last_cache_hit, 2)
        self.assertEqual(engine.cached_token_ids, [1, 2, 99, 65])

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
