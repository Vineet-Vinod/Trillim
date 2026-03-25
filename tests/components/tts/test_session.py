"""Tests for TTSSession behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.tts import TTSSession
from trillim.components.tts.public import TTS
from trillim.errors import SessionBusyError
from tests.components.tts.support import make_started_tts


class TTSSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self.spool_dir = Path(self._temp_dir.name) / "spool"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def _start_tts(self) -> TTS:
        tts, imports_patch, builtins_patch = make_started_tts()
        tts._spool_dir = self.spool_dir
        with patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root), builtins_patch, imports_patch:
            await tts.start()
        return tts

    def test_public_session_cannot_be_constructed_or_subclassed(self):
        with self.assertRaisesRegex(TypeError, "cannot be constructed directly"):
            TTSSession()
        with self.assertRaisesRegex(TypeError, "cannot be subclassed publicly"):
            type("BadSession", (TTSSession,), {})

    async def test_collect_and_iteration_are_mutually_exclusive(self):
        tts = await self._start_tts()
        session = await tts.speak("one two three")
        iterator = session.__aiter__()
        with self.assertRaises(SessionBusyError):
            await session.collect()
        chunks = [chunk async for chunk in iterator]
        self.assertTrue(chunks)
        await tts.stop()

    async def test_double_iteration_is_single_consumer(self):
        tts = await self._start_tts()
        session = await tts.speak("one two three")
        iterator = session.__aiter__()
        with self.assertRaises(SessionBusyError):
            session.__aiter__()
        self.assertTrue([chunk async for chunk in iterator])
        await tts.stop()
