"""Managed subprocess helpers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Mapping


class ManagedSubprocess:
    """Minimal lifecycle wrapper around an asyncio subprocess."""

    def __init__(
        self,
        *command: str,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        stdin: int | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> None:
        """Configure a subprocess to be started later."""
        if not command:
            raise ValueError("ManagedSubprocess requires at least one command element")
        self._command = tuple(command)
        self._cwd = None if cwd is None else str(Path(cwd))
        self._env = None if env is None else dict(env)
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self.process: asyncio.subprocess.Process | None = None

    @property
    def command(self) -> tuple[str, ...]:
        """Return the configured command tuple."""
        return self._command

    async def start(self) -> asyncio.subprocess.Process:
        """Start the subprocess if it is not already running."""
        if self.process is not None and self.process.returncode is None:
            return self.process
        env = None
        if self._env is not None:
            env = os.environ.copy()
            env.update(self._env)
        self.process = await asyncio.create_subprocess_exec(
            *self._command,
            cwd=self._cwd,
            env=env,
            stdin=self._stdin,
            stdout=self._stdout,
            stderr=self._stderr,
        )
        return self.process

    async def stop(self, *, kill_after: float = 1.0) -> int:
        """Terminate the subprocess and kill it if it does not exit in time."""
        if kill_after <= 0:
            raise ValueError("kill_after must be > 0")
        if self.process is None:
            return 0
        if self.process.returncode is not None:
            return self.process.returncode
        self.process.terminate()
        try:
            return await asyncio.wait_for(self.process.wait(), timeout=kill_after)
        except asyncio.TimeoutError:
            self.process.kill()
            return await self.process.wait()

    async def __aenter__(self) -> asyncio.subprocess.Process:
        """Start the subprocess when entering an async context manager."""
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Stop the subprocess when leaving an async context manager."""
        await self.stop()

