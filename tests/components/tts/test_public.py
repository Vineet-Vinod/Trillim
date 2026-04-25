from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from trillim.components.tts import TTS, TTSSession
from trillim.components.tts._engine import TTSEngineCrashedError
from trillim.components.tts._voices import publish_custom_voice
from trillim.errors import ComponentLifecycleError, InvalidRequestError, SessionBusyError

from tests.components.tts.support import fake_voice_state, make_started_tts, patched_tts_environment


class PublicTTSTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self._stacks = []

    async def asyncTearDown(self) -> None:
        for stack in reversed(self._stacks):
            stack.close()
        self._temp_dir.cleanup()

    async def _start_tts(self):
        tts, engine, stack = await make_started_tts(self.voice_root)
        self._stacks.append(stack)
        return tts, engine

    async def test_open_session_requires_started_component(self):
        tts = TTS()
        with self.assertRaisesRegex(ComponentLifecycleError, "TTS is not running"):
            await tts.open_session()

    async def test_start_stop_are_idempotent(self):
        tts, engine = await self._start_tts()
        await tts.start()
        await tts.stop()
        await tts.stop()
        self.assertTrue(engine.started)
        self.assertTrue(engine.stopped)

    async def test_start_loads_valid_custom_voices_from_disk(self):
        await publish_custom_voice(
            self.voice_root,
            name="stored",
            voice_state=fake_voice_state(),
            existing_names={"alba", "marius"},
        )

        stack = patched_tts_environment(self.voice_root)
        self._stacks.append(stack)
        tts = TTS()
        await tts.start()
        try:
            self.assertEqual(await tts.list_voices(), ["alba", "marius", "stored"])
            async with await tts.open_session(voice="stored") as session:
                self.assertEqual(await session.collect("hello"), b"  hello")
        finally:
            await tts.stop()

    async def test_session_collect_and_stream_use_engine_state(self):
        tts, engine = await self._start_tts()
        try:
            async with await tts.open_session() as session:
                self.assertIsInstance(session, TTSSession)
                self.assertEqual(session.state, "idle")
                self.assertEqual(session.voice, "alba")
                pcm = await session.collect("hello")
                self.assertEqual(pcm, b"  hello")
                self.assertEqual(session.state, "done")

            async with await tts.open_session(voice="marius") as session:
                chunks = [chunk async for chunk in session.synthesize("one. two.")]
                self.assertGreaterEqual(len(chunks), 2)

            self.assertTrue(engine.started)
            self.assertGreaterEqual(len(engine.synthesize_calls), 3)
            self.assertEqual(engine.synthesize_calls[0][1], "alba")
            self.assertEqual(engine.synthesize_calls[-1][1], "marius")
        finally:
            await tts.stop()

    async def test_session_can_change_voice_after_synthesis_finishes(self):
        tts, engine = await self._start_tts()
        try:
            session = await tts.open_session()
            self.assertEqual(await session.collect("hello"), b"  hello")
            await session.set_voice("marius")
            self.assertEqual(session.voice, "marius")
            self.assertEqual(await session.collect("again"), b"  again")
            self.assertEqual(engine.synthesize_calls[-1][1], "marius")
        finally:
            await tts.stop()

    async def test_session_propagates_synthesis_errors(self):
        tts, engine = await self._start_tts()
        try:
            engine.synthesize_error = RuntimeError("engine boom")
            session = await tts.open_session()
            with self.assertRaisesRegex(RuntimeError, "engine boom"):
                await session.collect("hello")
            self.assertEqual(session.state, "idle")
        finally:
            await tts.stop()

    async def test_session_rejects_concurrent_synthesis_and_voice_change(self):
        tts, engine = await self._start_tts()
        try:
            engine.synthesize_delay = 0.05
            session = await tts.open_session()
            task = asyncio.create_task(session.collect("one. two. three."))
            while session.state != "running":
                await asyncio.sleep(0)
            with self.assertRaisesRegex(SessionBusyError, "already synthesizing"):
                await session.collect("other")
            with self.assertRaisesRegex(SessionBusyError, "cannot change voice"):
                await session.set_voice("marius")
            await task
            await session.close()
        finally:
            await tts.stop()

    async def test_set_speed_is_best_effort_for_active_session(self):
        tts, _engine = await self._start_tts()
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
        tts, engine = await self._start_tts()
        try:
            engine.synthesize_delay = 0.05
            session = await tts.open_session()
            stream = session.synthesize(("word " * 1_000).strip())
            self.assertTrue(await stream.__anext__())
            await session.pause()

            next_chunk = asyncio.create_task(stream.__anext__())
            await asyncio.sleep(0.05)
            self.assertFalse(next_chunk.done())

            await session.resume()
            self.assertTrue(await asyncio.wait_for(next_chunk, timeout=1.0))
            await session.close()
        finally:
            await tts.stop()

    async def test_close_cancels_current_synthesis_and_session_can_be_reused(self):
        tts, engine = await self._start_tts()
        try:
            engine.synthesize_delay = 0.1
            session = await tts.open_session()
            task = asyncio.create_task(session.collect("one. two. three."))
            while session.state != "running":
                await asyncio.sleep(0)

            await session.close()
            self.assertEqual(await task, b"")
            self.assertEqual(session.state, "done")

            engine.synthesize_delay = 0
            self.assertEqual(await session.collect("again"), b"  again")
        finally:
            await tts.stop()

    async def test_register_list_delete_voice_updates_runtime_cache(self):
        tts, engine = await self._start_tts()
        try:
            self.assertEqual(await tts.list_voices(), ["alba", "marius"])
            self.assertEqual(await tts.register_voice("custom", b"voice"), "custom")
            self.assertEqual(await tts.list_voices(), ["alba", "marius", "custom"])
            async with await tts.open_session(voice="custom") as session:
                self.assertEqual(await session.collect("hello"), b"  hello")
            self.assertIsInstance(engine.synthesize_calls[-1][1], dict)
            self.assertEqual(await tts.delete_voice("custom"), "custom")
            self.assertEqual(await tts.list_voices(), ["alba", "marius"])
            with self.assertRaisesRegex(InvalidRequestError, "unknown voice"):
                await tts.open_session(voice="custom")
        finally:
            await tts.stop()

    async def test_register_voice_accepts_filesystem_paths(self):
        tts, engine = await self._start_tts()
        try:
            source = Path(self._temp_dir.name) / "voice.wav"
            source.write_bytes(b"voice")
            self.assertEqual(await tts.register_voice("frompath", source), "frompath")
            self.assertEqual(await tts.register_voice("fromstr", str(source)), "fromstr")
            self.assertEqual(len(engine.voice_build_calls), 2)
        finally:
            await tts.stop()

    async def test_register_voice_maps_client_build_errors(self):
        tts, engine = await self._start_tts()
        try:
            engine.build_error = TTSEngineCrashedError("unsupported audio input")
            with self.assertRaisesRegex(InvalidRequestError, "unsupported audio input"):
                await tts.register_voice("custom", b"voice")
        finally:
            await tts.stop()

    async def test_register_rejects_duplicate_and_delete_rejects_invalid_names(self):
        tts, _engine = await self._start_tts()
        try:
            with self.assertRaisesRegex(InvalidRequestError, "already exists"):
                await tts.register_voice("alba", b"voice")
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

    async def test_session_created_before_stop_yields_no_audio_after_stop(self):
        tts, _engine = await self._start_tts()
        session = await tts.open_session()
        await tts.stop()
        self.assertEqual(await session.collect("hello"), b"")
        self.assertEqual(session.state, "idle")


class EventLoopOwnershipTests(unittest.TestCase):
    def test_tts_is_bound_to_one_event_loop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "voices"

            async def start_component():
                tts, _engine, stack = await make_started_tts(root)
                return tts, stack

            tts, stack = asyncio.run(start_component())
            self.addCleanup(stack.close)

            async def use_from_new_loop():
                await tts.list_voices()

            with self.assertRaisesRegex(ComponentLifecycleError, "one event loop"):
                asyncio.run(use_from_new_loop())
