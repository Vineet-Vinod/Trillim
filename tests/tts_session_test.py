# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for TTS session queueing, interruption, and flow control."""

import asyncio
import struct
import unittest
from collections import deque

from trillim import TTS


def _pcm_silence(sample_count: int, *, sample: int = 0) -> bytes:
    return b"".join(struct.pack("<h", sample) for _ in range(sample_count))


class _SessionEngine:
    def __init__(self, plans, gates=None):
        self.sample_rate = 24000
        self.speed = 1.0
        self.calls: list[tuple[str, str | None]] = []
        self._plans = plans
        self._gates = gates or {}
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True

    async def _synthesize_raw_stream(
        self,
        text: str,
        voice: str | None = None,
    ):
        self.calls.append((text, voice))
        for index, chunk in enumerate(self._plans[text]):
            gate = self._gates.get((text, index))
            if gate is not None:
                await gate.wait()
            await asyncio.sleep(0)
            yield chunk


class _CleanupBlockingIterator:
    def __init__(self, cleanup_started: asyncio.Event, release_cleanup: asyncio.Event):
        self._cleanup_started = cleanup_started
        self._release_cleanup = release_cleanup

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(1)
        return _pcm_silence(64, sample=1)

    async def aclose(self):
        self._cleanup_started.set()
        await self._release_cleanup.wait()


class _StopOrderEngine:
    def __init__(self, cleanup_started: asyncio.Event, release_cleanup: asyncio.Event):
        self.sample_rate = 24000
        self.speed = 1.0
        self.cleanup_started = cleanup_started
        self.release_cleanup = release_cleanup
        self.stop_called = False

    async def stop(self) -> None:
        self.stop_called = True

    def _synthesize_raw_stream(
        self,
        text: str,
        voice: str | None = None,
    ):
        return _CleanupBlockingIterator(self.cleanup_started, self.release_cleanup)


class TTSSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop = asyncio.get_running_loop()

    def _make_tts(self, engine: _SessionEngine) -> TTS:
        tts = TTS()
        tts._engine = engine
        tts._loop = self.loop
        tts._active_session = None
        tts._queued_sessions = deque()
        return tts

    async def test_speak_queues_sessions_until_active_completes(self):
        release_first = asyncio.Event()
        first_chunk = _pcm_silence(256)
        second_chunk = _pcm_silence(256, sample=10)
        queued_chunk = _pcm_silence(128, sample=20)
        engine = _SessionEngine(
            {
                "first": [first_chunk, second_chunk],
                "second": [queued_chunk],
            },
            gates={("first", 1): release_first},
        )
        tts = self._make_tts(engine)

        first = tts.speak("first")
        second = tts.speak("second")

        self.assertEqual(first.state, "running")
        self.assertEqual(second.state, "queued")
        first_iter = first.__aiter__()
        self.assertEqual(await anext(first_iter), first_chunk)
        self.assertEqual(second.state, "queued")

        release_first.set()
        self.assertEqual([chunk async for chunk in first_iter], [second_chunk])
        self.assertEqual(await second.collect(), queued_chunk)
        self.assertEqual(
            engine.calls,
            [
                ("first", None),
                ("second", None),
            ],
        )

    async def test_speak_snapshots_explicit_speed_per_session(self):
        raw_chunk = _pcm_silence(4096)
        engine = _SessionEngine({"hello": [raw_chunk]})
        tts = self._make_tts(engine)

        session = tts.speak("hello", speed=2.0)
        audio = await session.collect()

        self.assertLess(len(audio), len(raw_chunk))
        self.assertEqual(session.speed, 2.0)
        self.assertEqual(engine.calls, [("hello", None)])

    async def test_pause_and_resume_gate_future_chunk_production(self):
        release_second = asyncio.Event()
        first_chunk = _pcm_silence(256)
        second_chunk = _pcm_silence(256, sample=30)
        engine = _SessionEngine(
            {"hello": [first_chunk, second_chunk]},
            gates={("hello", 1): release_second},
        )
        tts = self._make_tts(engine)
        session = tts.speak("hello")
        iterator = session.__aiter__()

        self.assertEqual(await anext(iterator), first_chunk)
        session.pause()
        await asyncio.sleep(0)
        self.assertEqual(session.state, "paused")

        next_chunk = asyncio.create_task(iterator.__anext__())
        release_second.set()
        await asyncio.sleep(0.01)
        self.assertFalse(next_chunk.done())

        session.resume()
        self.assertEqual(await next_chunk, second_chunk)
        await session.wait()
        self.assertEqual(session.state, "completed")

    async def test_session_backpressures_when_buffer_is_full(self):
        first_chunk = _pcm_silence(256, sample=31)
        second_chunk = _pcm_silence(256, sample=32)
        engine = _SessionEngine({"hello": [first_chunk, second_chunk]})
        tts = self._make_tts(engine)
        session = tts.speak("hello")
        session._chunks = asyncio.Queue(maxsize=2)
        session._chunk_slots = asyncio.Semaphore(1)

        wait_task = asyncio.create_task(session.wait())
        await asyncio.sleep(0.01)

        self.assertEqual(session._chunks.qsize(), 1)
        self.assertFalse(wait_task.done())

        iterator = session.__aiter__()
        self.assertEqual(await anext(iterator), first_chunk)
        await asyncio.wait_for(wait_task, timeout=0.5)
        self.assertEqual(await anext(iterator), second_chunk)
        with self.assertRaises(StopAsyncIteration):
            await iterator.__anext__()

    async def test_set_speed_changes_future_running_chunks(self):
        release_second = asyncio.Event()
        first_chunk = _pcm_silence(4096)
        second_chunk = _pcm_silence(4096, sample=50)
        engine = _SessionEngine(
            {"hello": [first_chunk, second_chunk]},
            gates={("hello", 1): release_second},
        )
        tts = self._make_tts(engine)
        session = tts.speak("hello")
        iterator = session.__aiter__()

        self.assertEqual(await anext(iterator), first_chunk)
        session.set_speed(2.0)
        await asyncio.sleep(0)
        release_second.set()
        remaining = b"".join([chunk async for chunk in iterator])

        self.assertLess(len(remaining), len(second_chunk))
        self.assertEqual(session.speed, 2.0)

    async def test_set_speed_updates_queued_session_before_start(self):
        release_active = asyncio.Event()
        active_chunk = _pcm_silence(256, sample=60)
        queued_chunk = _pcm_silence(4096, sample=70)
        engine = _SessionEngine(
            {
                "active": [active_chunk],
                "queued": [queued_chunk],
            },
            gates={("active", 0): release_active},
        )
        tts = self._make_tts(engine)

        active = tts.speak("active")
        queued = tts.speak("queued")
        queued.set_speed(0.5)
        await asyncio.sleep(0)
        self.assertEqual(queued.speed, 0.5)

        release_active.set()
        self.assertEqual(await active.collect(), active_chunk)
        queued_audio = await queued.collect()

        self.assertGreater(len(queued_audio), len(queued_chunk))

    async def test_set_speed_rejects_invalid_values(self):
        engine = _SessionEngine({"hello": [_pcm_silence(64)]})
        tts = self._make_tts(engine)
        session = tts.speak("hello")

        with self.assertRaisesRegex(ValueError, "speed must be between 0.25 and 4.0"):
            session.set_speed(5.0)

    async def test_interrupt_cancels_active_and_queued_sessions(self):
        block_active = asyncio.Event()
        active_chunk = _pcm_silence(256, sample=80)
        queued_chunk = _pcm_silence(128, sample=90)
        replacement_chunk = _pcm_silence(128, sample=100)
        engine = _SessionEngine(
            {
                "active": [active_chunk],
                "queued": [queued_chunk],
                "replacement": [replacement_chunk],
            },
            gates={("active", 0): block_active},
        )
        tts = self._make_tts(engine)

        active = tts.speak("active")
        queued = tts.speak("queued")
        replacement = tts.speak("replacement", interrupt=True)
        await asyncio.sleep(0)

        await active.wait()
        self.assertEqual(active.state, "cancelled")
        self.assertEqual(queued.state, "cancelled")
        self.assertEqual(await replacement.collect(), replacement_chunk)

    async def test_cancel_removes_queued_session(self):
        block_active = asyncio.Event()
        active_chunk = _pcm_silence(256, sample=110)
        queued_chunk = _pcm_silence(128, sample=120)
        engine = _SessionEngine(
            {
                "active": [active_chunk],
                "queued": [queued_chunk],
            },
            gates={("active", 0): block_active},
        )
        tts = self._make_tts(engine)

        active = tts.speak("active")
        queued = tts.speak("queued")
        queued.cancel()
        await asyncio.sleep(0)
        self.assertEqual(queued.state, "cancelled")

        block_active.set()
        self.assertEqual(await active.collect(), active_chunk)
        self.assertEqual(await queued.collect(), b"")

    async def test_session_timeout_marks_failure(self):
        never_release = asyncio.Event()
        engine = _SessionEngine(
            {"slow": [_pcm_silence(256, sample=5)]},
            gates={("slow", 0): never_release},
        )
        tts = self._make_tts(engine)

        session = tts.speak("slow", timeout=0.01)
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            await session.wait()
        self.assertEqual(session.state, "failed")

    async def test_stop_cancels_sessions_and_stops_engine(self):
        never_release = asyncio.Event()
        engine = _SessionEngine(
            {
                "slow": [_pcm_silence(256, sample=6)],
                "queued": [_pcm_silence(128, sample=7)],
            },
            gates={("slow", 0): never_release},
        )
        tts = self._make_tts(engine)

        session = tts.speak("slow")
        queued = tts.speak("queued")
        await tts.stop()

        self.assertTrue(session.done)
        self.assertEqual(session.state, "cancelled")
        self.assertTrue(queued.done)
        self.assertEqual(queued.state, "cancelled")
        self.assertIsNone(tts.engine)
        self.assertIsNone(tts._loop)
        self.assertTrue(engine.stopped)

    async def test_stop_waits_for_active_session_cleanup_before_engine_stop(self):
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        engine = _StopOrderEngine(cleanup_started, release_cleanup)
        tts = self._make_tts(engine)

        session = tts.speak("slow")
        await asyncio.sleep(0)

        stop_task = asyncio.create_task(tts.stop())
        await cleanup_started.wait()
        await asyncio.sleep(0.01)

        self.assertFalse(stop_task.done())
        self.assertFalse(engine.stop_called)

        release_cleanup.set()
        await stop_task

        self.assertTrue(engine.stop_called)
        self.assertEqual(session.state, "cancelled")
        self.assertIsNone(tts.engine)


if __name__ == "__main__":
    unittest.main()
