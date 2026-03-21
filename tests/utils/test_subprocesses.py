"""Tests for managed subprocess helpers."""

from __future__ import annotations

import asyncio
import sys
import unittest

from trillim.utils.subprocesses import ManagedSubprocess


class ManagedSubprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_command(self):
        with self.assertRaisesRegex(ValueError, "at least one command"):
            ManagedSubprocess()

    async def test_start_and_stop(self):
        proc = ManagedSubprocess(
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        )
        started = await proc.start()
        self.assertIs(proc.process, started)
        self.assertIsNone(started.returncode)
        code = await proc.stop(kill_after=0.01)
        self.assertIsInstance(code, int)
        self.assertIsNotNone(proc.process.returncode)

    async def test_rejects_non_positive_kill_timeout(self):
        proc = ManagedSubprocess(sys.executable, "-c", "print('ok')")
        await proc.start()
        with self.assertRaisesRegex(ValueError, "kill_after must be > 0"):
            await proc.stop(kill_after=0)
        await proc.stop()

