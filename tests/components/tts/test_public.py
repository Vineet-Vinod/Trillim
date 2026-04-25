from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from trillim.components.tts import TTS, TTSSession
from trillim.components.tts._engine import TTSEngine
from trillim.components.tts._session import _TTSSession
from trillim.components.tts._voices import publish_custom_voice
from trillim.errors import ComponentLifecycleError, InvalidRequestError, SessionBusyError

from tests.components.tts.support import (
    make_started_tts,
    reference_wav_bytes,
    tts_voice_store_environment,
    write_reference_wav,
)


class PublicTTSTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self._stacks = []

    async def asyncTearDown(self) -> None:
        for stack in reversed(self._stacks):
            stack.close()
        self._temp_dir.cleanup()

    async def _start_tts(self) -> TTS:
        tts, stack = await make_started_tts(self.voice_root)
        self._stacks.append(stack)
        return tts

    async def _build_real_voice_state(self) -> dict:
        source_path = Path(self._temp_dir.name) / "voice.wav"
        write_reference_wav(source_path)
        engine = TTSEngine()
        await engine.start()
        try:
            return await engine.build_voice_state(source_path)
        finally:
            await engine.stop()

    async def test_open_session_requires_started_component(self):
        tts = TTS()
        with self.assertRaisesRegex(ComponentLifecycleError, "TTS is not running"):
            await tts.open_session()

    async def test_start_stop_are_idempotent(self):
        tts = await self._start_tts()
        await tts.start()
        await tts.stop()
        await tts.stop()

    async def test_start_loads_valid_custom_voices_from_disk(self):
        voice_state = await self._build_real_voice_state()
        await publish_custom_voice(
            self.voice_root,
            name="stored",
            voice_state=voice_state,
            existing_names={"alba"},
        )

        stack = tts_voice_store_environment(self.voice_root)
        self._stacks.append(stack)
        tts = TTS()
        await tts.start()
        try:
            voices = await tts.list_voices()
            self.assertIn("alba", voices)
            self.assertIn("stored", voices)
            async with await tts.open_session(voice="stored") as session:
                self.assertEqual(session.voice, "stored")
        finally:
            await tts.stop()

    async def test_session_collect_and_stream_use_real_engine(self):
        tts = await self._start_tts()
        try:
            async with await tts.open_session() as session:
                self.assertIsInstance(session, TTSSession)
                self.assertEqual(session.state, "idle")
                self.assertEqual(session.voice, "alba")
                self._assert_pcm(await session.collect("hello"))
                self.assertEqual(session.state, "done")

            async with await tts.open_session(voice="marius") as session:
                chunks = [chunk async for chunk in session.synthesize("one. two.")]
                self.assertGreaterEqual(len(chunks), 1)
                for chunk in chunks:
                    self._assert_pcm(chunk)
        finally:
            await tts.stop()

    async def test_session_can_change_voice_after_synthesis_finishes(self):
        tts = await self._start_tts()
        try:
            session = await tts.open_session()
            self._assert_pcm(await session.collect("hello"))
            await session.set_voice("marius")
            self.assertEqual(session.voice, "marius")
            self._assert_pcm(await session.collect("again"))
        finally:
            await tts.stop()

    async def test_session_rejects_concurrent_synthesis_and_voice_change(self):
        tts = await self._start_tts()
        try:
            session = await tts.open_session()
            task = asyncio.create_task(session.collect("one. two. three. four. five."))
            while session.state != "running":
                await asyncio.sleep(0)
            with self.assertRaisesRegex(SessionBusyError, "already synthesizing"):
                await session.collect("other")
            with self.assertRaisesRegex(SessionBusyError, "cannot change voice"):
                await session.set_voice("marius")
            self._assert_pcm(await task)
            await session.close()
        finally:
            await tts.stop()

    async def test_set_speed_is_best_effort_for_active_session(self):
        tts = await self._start_tts()
        try:
            session = await tts.open_session()
            await session.set_speed(1.5)
            self.assertEqual(session.speed, 1.5)
            with self.assertRaisesRegex(InvalidRequestError, "speed"):
                await session.set_speed(99)
            await session.close()
        finally:
            await tts.stop()

    async def test_pause_blocks_consumer_delivery_until_resume(self):
        tts = await self._start_tts()
        try:
            session = await tts.open_session()
            stream = session.synthesize("one. two. three.")
            self._assert_pcm(await stream.__anext__())
            await session.pause()

            next_chunk = asyncio.create_task(stream.__anext__())
            await asyncio.sleep(0.05)
            self.assertFalse(next_chunk.done())

            await session.resume()
            self._assert_pcm(await asyncio.wait_for(next_chunk, timeout=30.0))
            await session.close()
        finally:
            await tts.stop()

    async def test_close_cancels_current_synthesis_and_session_can_be_reused(self):
        tts = await self._start_tts()
        try:
            session = await tts.open_session()
            task = asyncio.create_task(session.collect("one. two. three. four. five."))
            while session.state != "running":
                await asyncio.sleep(0)

            await session.close()
            cancelled_audio = await task
            self.assertIsInstance(cancelled_audio, bytes)
            self.assertEqual(session.state, "done")

            self._assert_pcm(await session.collect("again"))
        finally:
            await tts.stop()

    async def test_register_list_delete_voice_updates_runtime_cache(self):
        tts = await self._start_tts()
        try:
            voices = await tts.list_voices()
            self.assertIn("alba", voices)
            self.assertNotIn("custom", voices)
            self.assertEqual(await tts.register_voice("custom", reference_wav_bytes()), "custom")
            self.assertIn("custom", await tts.list_voices())
            async with await tts.open_session(voice="custom") as session:
                self._assert_pcm(await session.collect("hello"))
            self.assertEqual(await tts.delete_voice("custom"), "custom")
            self.assertNotIn("custom", await tts.list_voices())
            with self.assertRaisesRegex(InvalidRequestError, "unknown voice"):
                await tts.open_session(voice="custom")
        finally:
            await tts.stop()

    async def test_custom_voice_can_be_replaced_by_delete_then_register(self):
        tts = await self._start_tts()
        try:
            self.assertEqual(await tts.register_voice("custom", reference_wav_bytes()), "custom")
            async with await tts.open_session(voice="custom") as session:
                self._assert_pcm(await session.collect("hello"))

            self.assertEqual(await tts.delete_voice("custom"), "custom")
            self.assertEqual(await tts.register_voice("custom", reference_wav_bytes()), "custom")
            self.assertIn("custom", await tts.list_voices())
            async with await tts.open_session(voice="custom") as session:
                self._assert_pcm(await session.collect("hello"))
        finally:
            await tts.stop()

    async def test_register_voice_accepts_filesystem_paths(self):
        tts = await self._start_tts()
        try:
            source = Path(self._temp_dir.name) / "voice.wav"
            write_reference_wav(source)
            self.assertEqual(await tts.register_voice("frompath", source), "frompath")
            self.assertEqual(await tts.register_voice("fromstr", str(source)), "fromstr")
        finally:
            await tts.stop()

    async def test_register_voice_maps_client_build_errors(self):
        tts = await self._start_tts()
        try:
            with self.assertRaisesRegex(InvalidRequestError, "audio"):
                await tts.register_voice("custom", b"not valid audio")
        finally:
            await tts.stop()

    async def test_register_rejects_duplicate_and_delete_rejects_invalid_names(self):
        tts = await self._start_tts()
        try:
            with self.assertRaisesRegex(InvalidRequestError, "already exists"):
                await tts.register_voice("alba", reference_wav_bytes())
            with self.assertRaisesRegex(InvalidRequestError, "built in"):
                await tts.delete_voice("alba")
            with self.assertRaises(KeyError):
                await tts.delete_voice("missing")
            with self.assertRaisesRegex(InvalidRequestError, "audio must be bytes"):
                await tts.register_voice("custom", object())
            with self.assertRaisesRegex(InvalidRequestError, "only letters and digits"):
                await tts.open_session(voice="bad-name")
        finally:
            await tts.stop()

    async def test_session_created_before_stop_raises_after_stop(self):
        tts = await self._start_tts()
        session = await tts.open_session()
        await tts.stop()
        with self.assertRaisesRegex(ComponentLifecycleError, "component has been stopped"):
            await session.collect("hello")
        self.assertEqual(session.state, "idle")

    async def test_direct_async_use_is_bound_to_one_event_loop(self):
        tts = await self._start_tts()
        try:
            with self.assertRaisesRegex(ComponentLifecycleError, "one event loop"):
                await asyncio.to_thread(asyncio.run, tts.list_voices())
        finally:
            await tts.stop()

    def _assert_pcm(self, pcm: bytes) -> None:
        self.assertIsInstance(pcm, bytes)
        self.assertGreater(len(pcm), 0)
        self.assertEqual(len(pcm) % 2, 0)

class SessionAndEngineContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_tts_session_cannot_be_constructed_directly(self):
        with self.assertRaises(TypeError):
            TTSSession()

    async def test_tts_session_cannot_be_subclassed_publicly(self):
        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class CustomSession(TTSSession):
                pass

    async def test_private_tts_session_requires_owner_token(self):
        tts = TTS()
        with self.assertRaisesRegex(TypeError, "open_session"):
            _TTSSession(tts)

    async def test_engine_public_lifecycle_contract(self):
        engine = TTSEngine()
        await engine.stop()
        with self.assertRaisesRegex(ComponentLifecycleError, "not running"):
            await engine.synthesize_segment("hello", voice_state="alba")
        await engine.start()
        try:
            await engine.start()
        finally:
            await engine.stop()
        await engine.stop()
