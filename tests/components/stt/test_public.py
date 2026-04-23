from __future__ import annotations

import asyncio
import io
import unittest
import wave
from pathlib import Path

import numpy as np

from trillim.components.stt import STT, STTSession
from trillim.components.stt._engine import STTEngine
from trillim.errors import ComponentLifecycleError, InvalidRequestError, SessionBusyError
from trillim.runtime import Runtime

EXPECTED_PHRASES = (
    "torpedo",
    "russian grand prix",
    "austrian grand prix",
    "british grand prix",
)


class PublicSTTTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.fixture_path = Path(__file__).with_name("test.wav")
        self.fixture_bytes = self.fixture_path.read_bytes()

    async def _start_stt(self) -> STT:
        stt = STT()
        await stt.start()
        return stt

    def _make_8bit_wav(self) -> bytes:
        with wave.open(str(self.fixture_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()
            rate = wav_file.getframerate()
        samples = np.frombuffer(frames, dtype="<i2")
        unsigned = np.clip(
            ((samples.astype(np.int32) + 32768) >> 8),
            0,
            255,
        ).astype(np.uint8)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(1)
            wav_file.setframerate(rate)
            wav_file.writeframes(unsigned.tobytes())
        return buffer.getvalue()

    def _make_empty_wav(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"")
        return buffer.getvalue()

    def _make_24bit_wav(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(3)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00\x00" * 16)
        return buffer.getvalue()

    async def test_open_session_requires_started_component(self):
        stt = STT()
        with self.assertRaisesRegex(ComponentLifecycleError, "STT is not running"):
            stt.open_session()

    async def test_transcribe_bytes_and_file_return_expected_text(self):
        stt = await self._start_stt()
        try:
            bytes_text = await stt.transcribe_bytes(self.fixture_bytes)
            file_text = await stt.transcribe_file(str(self.fixture_path))
            self.assertEqual(bytes_text, file_text)
            self._assert_expected_transcript(bytes_text)
        finally:
            await stt.stop()

    async def test_audio_session_context_manager_and_state_transitions(self):
        stt = await self._start_stt()
        try:
            async with stt.open_session() as session:
                self.assertEqual(session.state, "idle")
                self._assert_expected_transcript(await session.transcribe(self.fixture_bytes))
                self.assertEqual(session.state, "done")
        finally:
            await stt.stop()

    async def test_audio_session_rejects_concurrent_transcribe_calls(self):
        stt = await self._start_stt()
        try:
            session = stt.open_session()
            task = asyncio.create_task(session.transcribe(self.fixture_bytes))
            while session.state != "transcribing":
                await asyncio.sleep(0)
            with self.assertRaisesRegex(SessionBusyError, "already transcribing"):
                await session.transcribe(self.fixture_bytes)
            self._assert_expected_transcript(await task)
            self.assertEqual(session.state, "done")
        finally:
            await stt.stop()

    async def test_sdk_concurrent_transcriptions_each_return_expected_text(self):
        stt = await self._start_stt()
        try:
            results = await asyncio.gather(
                stt.transcribe_bytes(self.fixture_bytes),
                stt.transcribe_file(self.fixture_path),
            )
            self.assertEqual(results[0], results[1])
            self._assert_expected_transcript(results[0])
        finally:
            await stt.stop()

    async def test_session_created_before_stop_returns_empty_text_after_stop(self):
        stt = await self._start_stt()
        session = stt.open_session()
        await stt.stop()
        self.assertEqual(await session.transcribe(self.fixture_bytes), "")
        self.assertEqual(session.state, "done")

    async def test_truncated_wav_is_rejected(self):
        stt = await self._start_stt()
        try:
            with self.assertRaisesRegex(InvalidRequestError, "invalid WAV audio"):
                await stt.transcribe_bytes(self.fixture_bytes[:64])
        finally:
            await stt.stop()

    async def test_8bit_wav_is_accepted(self):
        stt = await self._start_stt()
        try:
            text = await stt.transcribe_bytes(self._make_8bit_wav())
            self.assertIsInstance(text, str)
            self.assertIn("torpedo", text.lower())
        finally:
            await stt.stop()

    async def test_transcribe_bytes_rejects_invalid_sdk_inputs(self):
        stt = await self._start_stt()
        try:
            with self.assertRaisesRegex(InvalidRequestError, "audio must be bytes"):
                await stt.transcribe_bytes("bad")  # type: ignore[arg-type]
            with self.assertRaisesRegex(InvalidRequestError, "audio must not be empty"):
                await stt.transcribe_bytes(b"")
            with self.assertRaisesRegex(InvalidRequestError, "whole 16-bit samples"):
                await stt.transcribe_bytes(b"\x00")
            with self.assertRaisesRegex(InvalidRequestError, "audio must not be empty"):
                await stt.transcribe_bytes(self._make_empty_wav())
            with self.assertRaisesRegex(InvalidRequestError, "unsupported WAV sample width"):
                await stt.transcribe_bytes(self._make_24bit_wav())
        finally:
            await stt.stop()

    async def test_transcribe_bytes_accepts_bytearray_and_memoryview(self):
        stt = await self._start_stt()
        try:
            bytearray_text = await stt.transcribe_bytes(bytearray(self.fixture_bytes))
            memoryview_text = await stt.transcribe_bytes(memoryview(self.fixture_bytes))
            self.assertEqual(bytearray_text, memoryview_text)
            self._assert_expected_transcript(bytearray_text)
        finally:
            await stt.stop()

    def _assert_expected_transcript(self, text: str) -> None:
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 100)
        lowered = text.lower()
        for phrase in EXPECTED_PHRASES:
            self.assertIn(phrase, lowered)


class RuntimeSTTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = Path(__file__).with_name("test.wav")
        self.fixture_bytes = self.fixture_path.read_bytes()

    def test_runtime_syncify_supports_component_and_session_usage(self):
        with Runtime(STT()) as runtime:
            file_text = runtime.stt.transcribe_file(self.fixture_path)
            self.assertIsInstance(file_text, str)
            self.assertGreater(len(file_text), 100)
            with runtime.stt.open_session() as session:
                self.assertEqual(session.state, "idle")
                self.assertEqual(session.transcribe(self.fixture_bytes), file_text)
                self.assertEqual(session.state, "done")


class SessionAndEngineContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_stt_session_cannot_be_constructed_directly(self):
        with self.assertRaises(TypeError):
            STTSession()

    async def test_engine_public_lifecycle_contract(self):
        engine = STTEngine()
        await engine.stop()
        with self.assertRaisesRegex(ComponentLifecycleError, "not running"):
            await engine.transcribe(b"\x00\x00")
        await engine.start()
        try:
            await engine.start()
        finally:
            await engine.stop()
        await engine.stop()
