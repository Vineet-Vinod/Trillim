"""Tests for the public TTS component API."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trillim.components.tts._limits import MAX_EMITTED_AUDIO_CHUNKS
from trillim.components.tts.public import TTS
from trillim.components.tts._worker import WorkerFailureError
from trillim.errors import (
    AdmissionRejectedError,
    InvalidRequestError,
    OperationCancelledError,
    SessionClosedError,
)
from tests.components.tts.support import make_started_tts


class PublicTTSTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self.spool_dir = Path(self._temp_dir.name) / "spool"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def _start_tts(self, **kwargs) -> TTS:
        tts, imports_patch, builtins_patch = make_started_tts(**kwargs)
        tts._spool_dir = self.spool_dir
        with patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root), builtins_patch, imports_patch:
            await tts.start()
        return tts

    async def _wait_until(self, predicate, *, timeout: float = 1.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                self.fail("timed out waiting for condition")
            await asyncio.sleep(0)

    async def test_start_requires_known_default_voice(self):
        tts, imports_patch, builtins_patch = make_started_tts(default_voice="missing")
        with patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root), builtins_patch, imports_patch:
            with self.assertRaisesRegex(ValueError, "unknown default_voice"):
                await tts.start()

    async def test_register_list_and_delete_voice(self):
        tts = await self._start_tts()
        self.assertEqual(await tts.list_voices(), ["alba", "marius"])
        self.assertEqual(await tts.register_voice("custom", b"voice"), "custom")
        self.assertEqual(await tts.list_voices(), ["alba", "marius", "custom"])
        self.assertEqual(await tts.delete_voice("custom"), "custom")
        await tts.stop()

    async def test_register_voice_accepts_string_and_path_inputs(self):
        tts = await self._start_tts()
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        second_source = Path(self._temp_dir.name) / "voice-2.wav"
        second_source.write_bytes(b"voice-2")
        self.assertEqual(await tts.register_voice("custom-path", source), "custom-path")
        self.assertEqual(await tts.register_voice("custom-str", str(second_source)), "custom-str")
        await tts.stop()

    async def test_register_voice_rejects_duplicate_names(self):
        tts = await self._start_tts()
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        self.assertEqual(await tts.register_voice("custom", str(source)), "custom")
        with self.assertRaisesRegex(Exception, "already exists"):
            await tts.register_voice("custom", str(source))
        await tts.stop()

    async def test_register_voice_maps_worker_failure_to_invalid_request(self):
        async def bad_voice_builder(_audio_path: Path) -> bytes:
            raise WorkerFailureError("unsupported or malformed audio input")

        tts = await self._start_tts(voice_state_builder=bad_voice_builder)
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        with self.assertRaisesRegex(InvalidRequestError, "unsupported or malformed audio input"):
            await tts.register_voice("custom", str(source))
        await tts.stop()

    async def test_register_voice_preserves_backend_worker_failure(self):
        async def bad_voice_builder(_audio_path: Path) -> bytes:
            raise WorkerFailureError("backend voice builder crashed")

        tts = await self._start_tts(voice_state_builder=bad_voice_builder)
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        with self.assertRaisesRegex(WorkerFailureError, "backend voice builder crashed"):
            await tts.register_voice("custom", str(source))
        await tts.stop()

    async def test_speak_collect_and_synthesize_wav(self):
        tts = await self._start_tts()
        session = await tts.speak("hello world")
        pcm = await session.collect()
        self.assertIn(b"hello world", pcm)
        wav = await tts.synthesize_wav("tiny prompt")
        self.assertTrue(wav.startswith(b"RIFF"))
        await tts.stop()

    async def test_speak_with_speed_change_uses_multi_chunk_stretching_without_crashing(self):
        pcm_chunk = (b"\x00\x10" * 4_800)

        async def pcm_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            return pcm_chunk

        tts = await self._start_tts(synth=pcm_synth)
        text = " ".join(f"word{index}" for index in range(60))
        session = await tts.speak(text, speed=2.0)
        pcm = await session.collect()
        self.assertTrue(pcm)
        self.assertEqual(len(pcm) % 2, 0)
        await tts.stop()

    async def test_second_request_is_rejected_while_running(self):
        started = asyncio.Event()
        unblock = asyncio.Event()

        async def blocking_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            started.set()
            await unblock.wait()
            return b"pcm"

        tts = await self._start_tts(synth=blocking_synth)
        active = await tts.speak("one two three four five")
        await started.wait()
        self.assertEqual(active.state, "running")
        with self.assertRaisesRegex(AdmissionRejectedError, "only one live session"):
            await tts.speak("overflow")
        unblock.set()
        await asyncio.wait_for(active.collect(), timeout=1)
        await asyncio.wait_for(tts.stop(), timeout=1)

    async def test_stale_reservation_cannot_start_a_session(self):
        tts = await self._start_tts()
        reservation = await tts._reserve_session_slot()
        await tts._release_reserved_slot(reservation)
        with self.assertRaisesRegex(AdmissionRejectedError, "reservation is no longer active"):
            await tts._start_reserved_session(
                reservation,
                "hello world",
                voice="alba",
                speed=1.0,
            )
        followup = await tts.speak("hello again")
        self.assertTrue(await asyncio.wait_for(followup.collect(), timeout=1))
        await tts.stop()

    async def test_reserved_slot_blocks_followup_requests(self):
        tts = await self._start_tts()
        reservation = await tts._reserve_session_slot()
        with self.assertRaisesRegex(AdmissionRejectedError, "only one live session"):
            await tts.speak("hello world")
        await tts._release_reserved_slot(reservation)
        await tts.stop()

    async def test_failed_reserved_start_cleans_up_temporary_voice_copy(self):
        tts = await self._start_tts()
        reservation = await tts._reserve_session_slot()
        await tts._release_reserved_slot(reservation)
        cleanup_path = self.spool_dir / "voice.state"
        cleanup_path.parent.mkdir(parents=True, exist_ok=True)
        cleanup_path.write_bytes(b"voice")
        resolved_voice = SimpleNamespace(
            kind="custom",
            reference="voice-ref",
            cleanup_path=cleanup_path,
        )

        async def resolve_for_session(_voice: str, *, spool_dir: Path):
            self.assertEqual(spool_dir, self.spool_dir)
            return resolved_voice

        with patch.object(
            tts._voice_store,
            "resolve_for_session",
            side_effect=resolve_for_session,
        ):
            with self.assertRaisesRegex(AdmissionRejectedError, "reservation is no longer active"):
                await tts._start_reserved_session(
                    reservation,
                    "hello world",
                    voice="custom",
                    speed=1.0,
                )
        self.assertFalse(cleanup_path.exists())
        await tts.stop()

    async def test_stop_clears_reserved_slot(self):
        tts = await self._start_tts()
        reservation = await tts._reserve_session_slot()
        self.assertIs(tts._reserved_slot, reservation)
        await tts.stop()
        self.assertIsNone(tts._reserved_slot)

    async def test_second_request_is_rejected_while_paused(self):
        first_segment_started = asyncio.Event()
        first_segment_release = asyncio.Event()
        synth_calls = 0

        async def gated_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            nonlocal synth_calls
            synth_calls += 1
            if synth_calls == 1:
                first_segment_started.set()
                await first_segment_release.wait()
            return b"pcm"

        tts = await self._start_tts(synth=gated_synth)
        text = " ".join(f"word{index}" for index in range(30))
        session = await tts.speak(text)
        await first_segment_started.wait()
        await session.pause()
        first_segment_release.set()
        await self._wait_until(lambda: session.state == "paused")
        self.assertEqual(session.state, "paused")
        with self.assertRaisesRegex(AdmissionRejectedError, "only one live session"):
            await tts.speak("other words")
        await session.cancel()
        await tts.stop()

    async def test_second_request_is_rejected_while_prior_worker_is_still_closing(self):
        close_started = asyncio.Event()
        allow_close = asyncio.Event()

        tts = await self._start_tts()
        session = await tts.speak("hello world")
        original_close = session._session_worker.close

        async def delayed_close() -> None:
            close_started.set()
            await allow_close.wait()
            await original_close()

        session._session_worker.close = delayed_close
        collect_task = asyncio.create_task(session.collect())
        await close_started.wait()
        with self.assertRaisesRegex(AdmissionRejectedError, "only one live session"):
            await tts.speak("other words")
        allow_close.set()
        await asyncio.wait_for(collect_task, timeout=1)
        next_session = await tts.speak("other words")
        self.assertTrue(await asyncio.wait_for(next_session.collect(), timeout=1))
        await tts.stop()

    async def test_worker_close_failure_finishes_session_and_frees_slot(self):
        tts = await self._start_tts()
        session = await tts.speak("hello world")

        async def failing_close() -> None:
            raise RuntimeError("worker close failed")

        session._session_worker.close = failing_close
        with self.assertRaisesRegex(RuntimeError, "worker close failed"):
            await session.collect()
        self.assertEqual(session.state, "failed")
        next_session = await tts.speak("other words")
        self.assertTrue(await asyncio.wait_for(next_session.collect(), timeout=1))
        await tts.stop()

    async def test_pause_resume_and_cancel_work_across_safe_boundaries(self):
        seen_texts: list[str] = []
        first_segment_started = asyncio.Event()
        first_segment_release = asyncio.Event()

        async def blocking_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del voice_kind, voice_reference
            seen_texts.append(text)
            if len(seen_texts) == 1:
                first_segment_started.set()
                await first_segment_release.wait()
            return text.encode("utf-8")

        tts = await self._start_tts(synth=blocking_synth)
        first = await tts.speak(
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
            "twenty twentyone twentytwo"
        )
        await first_segment_started.wait()
        await first.pause()
        first_segment_release.set()
        await self._wait_until(lambda: first.state == "paused")
        self.assertEqual(first.state, "paused")
        self.assertEqual(len(seen_texts), 1)
        await first.resume()
        first_pcm = await asyncio.wait_for(first.collect(), timeout=1)
        self.assertEqual(first.state, "completed")
        self.assertTrue(first_pcm)
        self.assertGreaterEqual(len(seen_texts), 2)
        await asyncio.wait_for(tts.stop(), timeout=1)

    async def test_cancel_active_session_propagates_operation_cancelled(self):
        started = asyncio.Event()

        async def hanging_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            started.set()
            await asyncio.Future()
            return b"never"

        tts = await self._start_tts(synth=hanging_synth)
        session = await tts.speak("hello there")
        consumer = asyncio.create_task(session.collect())
        await started.wait()
        await session.cancel()
        with self.assertRaises(OperationCancelledError):
            await consumer
        await tts.stop()

    async def test_stop_cancels_live_sessions(self):
        started = asyncio.Event()

        async def hanging_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            started.set()
            await asyncio.Future()
            return b"never"

        tts = await self._start_tts(synth=hanging_synth)
        session = await tts.speak("hello there")
        consumer = asyncio.create_task(session.collect())
        await started.wait()
        await tts.stop()
        with self.assertRaises(SessionClosedError):
            await consumer

    async def test_pause_then_cancel_under_backpressure_cancels_blocked_put(self):
        allow_progress = asyncio.Event()
        blocked_put = asyncio.Event()
        put_calls = 0

        async def fast_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            await allow_progress.wait()
            return b"pcm"

        tts = await self._start_tts(synth=fast_synth)
        text = " ".join(
            f"word{index}" for index in range((MAX_EMITTED_AUDIO_CHUNKS + 2) * 20)
        )
        session = await tts.speak(text)
        original_put_chunk = session._put_chunk

        async def tracking_put_chunk(chunk: bytes) -> None:
            nonlocal put_calls
            put_calls += 1
            if put_calls == MAX_EMITTED_AUDIO_CHUNKS + 1:
                blocked_put.set()
            await original_put_chunk(chunk)

        session._put_chunk = tracking_put_chunk
        allow_progress.set()
        await asyncio.wait_for(blocked_put.wait(), timeout=1)
        await self._wait_until(
            lambda: session._audio_queue.full()
            and session._task is not None
            and not session._task.done()
        )
        await session.pause()
        self.assertEqual(session.state, "running")
        await asyncio.wait_for(session.cancel(), timeout=1)
        self.assertIsNotNone(session._task)
        assert session._task is not None
        self.assertTrue(session._task.done())
        await tts.stop()

    async def test_tampered_custom_voice_store_fails_closed_but_builtin_tts_still_works(self):
        self.voice_root.mkdir(parents=True, exist_ok=True)
        (self.voice_root / "manifest.json").write_text(
            json.dumps(
                {
                    "voices": [
                        {
                            "name": "custom",
                            "storage_id": "../escape",
                            "size_bytes": 4,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        tts = await self._start_tts()
        pcm = await (await tts.speak("hello world")).collect()
        self.assertIn(b"hello world", pcm)
        with self.assertRaisesRegex(Exception, "tampered"):
            await tts.list_voices()
        with self.assertRaisesRegex(Exception, "tampered"):
            await tts.speak("hello again", voice="custom")
        await tts.stop()
