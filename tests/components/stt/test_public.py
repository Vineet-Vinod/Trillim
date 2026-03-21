"""Tests for the public STT component API."""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.stt._config import SourceFileSnapshot
from trillim.components.stt.public import STT
from trillim.errors import AdmissionRejectedError, InvalidRequestError, ProgressTimeoutError
from tests.components.stt.support import list_spool_files, make_faster_whisper_stub


class PublicSTTTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.spool_dir = Path(self._temp_dir.name) / "spool"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def _start_stt(self) -> STT:
        stt = STT()
        stt._spool_dir = self.spool_dir
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}):
            await stt.start()
        return stt

    async def test_start_fails_when_faster_whisper_is_missing(self):
        stt = STT()
        with patch(
            "trillim.components.stt.public.importlib.import_module",
            side_effect=ModuleNotFoundError("faster_whisper"),
        ):
            with self.assertRaises(ModuleNotFoundError):
                await stt.start()
        self.assertFalse(stt._started)

    async def test_transcribe_bytes_returns_text_and_cleans_up_owned_temp_file(self):
        stt = await self._start_stt()
        seen_path: dict[str, Path] = {}

        async def fake_worker(audio_path: Path, *, language: str | None) -> str:
            path = Path(audio_path)
            seen_path["path"] = path
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, self.spool_dir)
            self.assertEqual(language, "en")
            return "hello"

        with patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=fake_worker,
        ):
            text = await stt.transcribe_bytes(b"abc", language="en")

        self.assertEqual(text, "hello")
        self.assertFalse(seen_path["path"].exists())

    async def test_transcribe_file_copies_into_owned_temp_and_cleans_up(self):
        stt = await self._start_stt()
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"abc")
        seen_path: dict[str, Path] = {}

        async def fake_worker(audio_path: Path, *, language: str | None) -> str:
            path = Path(audio_path)
            seen_path["path"] = path
            self.assertNotEqual(path, source)
            self.assertEqual(path.read_bytes(), b"abc")
            return "hello"

        with patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=fake_worker,
        ):
            text = await stt.transcribe_file(source, language=None)

        self.assertEqual(text, "hello")
        self.assertFalse(seen_path["path"].exists())

    async def test_transcribe_file_rejects_changed_source_and_cleans_up(self):
        stt = await self._start_stt()
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"abc")
        with patch(
            "trillim.components.stt._spool.snapshot_source_file",
            side_effect=[
                SourceFileSnapshot(size_bytes=3, modified_ns=1),
                SourceFileSnapshot(size_bytes=3, modified_ns=2),
            ],
        ):
            with self.assertRaisesRegex(InvalidRequestError, "changed while it was being copied"):
                await stt.transcribe_file(source)
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_busy_byte_request_fails_before_spooling(self):
        stt = await self._start_stt()
        lease = await stt._admission.acquire()

        async def fail_if_called(*args, **kwargs):
            raise AssertionError("spool helper should not be called")

        with patch(
            "trillim.components.stt.public.spool_audio_bytes",
            side_effect=fail_if_called,
        ):
            with self.assertRaisesRegex(AdmissionRejectedError, "STT is busy"):
                await stt.transcribe_bytes(b"abc")
        await lease.release()

    async def test_busy_file_request_fails_before_worker_launch(self):
        stt = await self._start_stt()
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"abc")
        lease = await stt._admission.acquire()

        async def fail_if_called(*args, **kwargs):
            raise AssertionError("worker should not be called")

        with patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=fail_if_called,
        ):
            with self.assertRaisesRegex(AdmissionRejectedError, "STT is busy"):
                await stt.transcribe_file(source)
        await lease.release()

    async def test_stop_cancels_active_work_and_waits_for_cleanup(self):
        stt = await self._start_stt()
        started = asyncio.Event()
        seen_path: dict[str, Path] = {}

        async def hanging_worker(audio_path: Path, *, language: str | None) -> str:
            seen_path["path"] = Path(audio_path)
            started.set()
            await asyncio.Event().wait()
            return "never"

        with patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=hanging_worker,
        ):
            task = asyncio.create_task(stt.transcribe_bytes(b"abc"))
            await started.wait()
            await stt.stop()
            self.assertFalse(stt._started)
            self.assertFalse(seen_path["path"].exists())
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_stop_cancels_active_file_copy_and_waits_for_cleanup(self):
        stt = await self._start_stt()
        source = Path(self._temp_dir.name) / "source.wav"
        source.write_bytes(b"abcdef")
        started = threading.Event()
        original_read = __import__(
            "trillim.components.stt._spool",
            fromlist=["_read_source_chunk"],
        )._read_source_chunk

        def slow_read(source_handle):
            chunk = original_read(source_handle)
            if chunk and not started.is_set():
                started.set()
                time.sleep(0.05)
            return chunk

        with patch("trillim.components.stt._spool.SPOOL_CHUNK_SIZE_BYTES", 1), patch(
            "trillim.components.stt._spool._read_source_chunk",
            side_effect=slow_read,
        ):
            task = asyncio.create_task(stt.transcribe_file(source))
            await asyncio.to_thread(started.wait)
            await stt.stop()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(list_spool_files(self.spool_dir), [])

    async def test_timeout_error_surfaces_to_sdk_callers(self):
        stt = await self._start_stt()
        with patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=ProgressTimeoutError("timed out"),
        ):
            with self.assertRaisesRegex(ProgressTimeoutError, "timed out"):
                await stt.transcribe_bytes(b"abc")

    async def test_transcribe_requires_started_component(self):
        stt = STT()
        with self.assertRaisesRegex(RuntimeError, "not started"):
            await stt.transcribe_bytes(b"abc")
