"""Tests for the public TTS component API."""

from __future__ import annotations

import asyncio
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import APIRouter

from tests.components.tts.support import make_started_tts
from trillim.components.tts._limits import (
    MAX_EMITTED_AUDIO_CHUNKS,
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    TARGET_TTS_TOKENS,
)
from trillim.components.tts._session import _create_tts_session
from trillim.components.tts._worker import WorkerFailureError
from trillim.components.tts.public import (
    TTS,
    _apply_exponential_fade_in_pcm,
    _boundary_pause_ms,
    _pcm_silence,
    _postprocess_segment_pcm,
    _segment_pause_pcm,
    _StreamingPCMStretcher,
)
from trillim.errors import (
    AdmissionRejectedError,
    InvalidRequestError,
    OperationCancelledError,
    SessionClosedError,
)


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
        with (
            patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root),
            builtins_patch,
            imports_patch,
        ):
            await tts.start()
        return tts

    async def _wait_until(self, predicate, *, timeout: float = 1.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                self.fail("timed out waiting for condition")
            await asyncio.sleep(0)

    def _make_internal_session(self, tts: TTS):
        return _create_tts_session(
            tts,
            text="hello world",
            voice="alba",
            voice_kind="predefined",
            voice_reference="alba",
            speed=1.0,
            cleanup_path=None,
            session_worker=SimpleNamespace(close=lambda: None),
        )

    async def test_start_requires_known_default_voice(self):
        tts, imports_patch, builtins_patch = make_started_tts(default_voice="missing")
        with (
            patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root),
            builtins_patch,
            imports_patch,
        ):
            with self.assertRaisesRegex(ValueError, "unknown default_voice"):
                await tts.start()

    async def test_public_properties_router_and_idempotent_start(self):
        tts = await self._start_tts()

        self.assertEqual(tts.default_voice, "alba")
        self.assertEqual(tts.speed, 1.0)
        self.assertIsInstance(tts.router(), APIRouter)

        await tts.start()
        await tts.stop()

    async def test_start_accepts_existing_custom_default_voice(self):
        initial = await self._start_tts()
        await initial.register_voice("custom", b"voice")
        await initial.stop()

        tts, imports_patch, builtins_patch = make_started_tts(default_voice="custom")
        tts._spool_dir = self.spool_dir
        with (
            patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root),
            builtins_patch,
            imports_patch,
        ):
            await tts.start()
        await tts.stop()

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
        self.assertEqual(await tts.register_voice("custompath", source), "custompath")
        self.assertEqual(
            await tts.register_voice("customstr", str(second_source)), "customstr"
        )
        await tts.stop()

    async def test_register_voice_rejects_empty_path_objects(self):
        tts = await self._start_tts()
        with self.assertRaisesRegex(InvalidRequestError, "path is required"):
            await tts.register_voice("custom", Path(""))
        await tts.stop()

    async def test_register_voice_rejects_non_alphanumeric_names(self):
        tts = await self._start_tts()
        for name in ("bad/name", "../escape", "bad-name", "bad_name", "bad.name"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    InvalidRequestError,
                    "must contain only letters and digits",
                ):
                    await tts.register_voice(name, b"voice")
        await tts.stop()

    async def test_register_voice_rejects_duplicate_names(self):
        tts = await self._start_tts()
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        self.assertEqual(await tts.register_voice("custom", str(source)), "custom")
        with self.assertRaisesRegex(Exception, "already exists"):
            await tts.register_voice("custom", str(source))
        await tts.stop()

    async def test_register_voice_rejects_invalid_audio_types(self):
        tts = await self._start_tts()
        with self.assertRaisesRegex(
            InvalidRequestError, "audio must be bytes, str, or Path"
        ):
            await tts.register_voice("custom", 123)  # type: ignore[arg-type]
        await tts.stop()

    async def test_register_voice_maps_worker_failure_to_invalid_request(self):
        async def bad_voice_builder(_audio_path: Path) -> bytes:
            raise WorkerFailureError("unsupported or malformed audio input")

        tts = await self._start_tts(voice_state_builder=bad_voice_builder)
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        with self.assertRaisesRegex(
            InvalidRequestError, "unsupported or malformed audio input"
        ):
            await tts.register_voice("custom", str(source))
        await tts.stop()

    async def test_register_voice_maps_voice_state_size_limit_to_invalid_request(self):
        async def bad_voice_builder(_audio_path: Path) -> bytes:
            raise WorkerFailureError(
                "custom voice state exceeds the 64 MB limit; use a shorter reference sample"
            )

        tts = await self._start_tts(voice_state_builder=bad_voice_builder)
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        with self.assertRaisesRegex(
            InvalidRequestError,
            "custom voice state exceeds the 64 MB limit; use a shorter reference sample",
        ):
            await tts.register_voice("custom", str(source))
        await tts.stop()

    async def test_register_voice_preserves_backend_worker_failure(self):
        async def bad_voice_builder(_audio_path: Path) -> bytes:
            raise WorkerFailureError("backend voice builder crashed")

        tts = await self._start_tts(voice_state_builder=bad_voice_builder)
        source = Path(self._temp_dir.name) / "voice.wav"
        source.write_bytes(b"voice")
        with self.assertRaisesRegex(
            WorkerFailureError, "backend voice builder crashed"
        ):
            await tts.register_voice("custom", str(source))
        await tts.stop()

    async def test_register_voice_does_not_report_failure_after_publish_when_upload_cleanup_fails(
        self,
    ):
        tts = await self._start_tts()
        original_unlink = Path.unlink

        def fail_owned_upload_cleanup(path: Path, *args, **kwargs):
            if (
                path.parent == self.spool_dir
                and path.name.startswith("tts-")
                and path.suffix == ".audio"
            ):
                raise PermissionError("cleanup boom")
            return original_unlink(path, *args, **kwargs)

        try:
            with patch(
                "pathlib.Path.unlink",
                autospec=True,
                side_effect=fail_owned_upload_cleanup,
            ):
                self.assertEqual(await tts.register_voice("custom", b"voice"), "custom")
        finally:
            await tts.stop()

    async def test_speak_collect_and_synthesize_wav(self):
        tts = await self._start_tts()
        session = await tts.speak("hello world")
        pcm = await session.collect()
        self.assertTrue(pcm)
        wav = await tts.synthesize_wav("tiny prompt")
        self.assertTrue(wav.startswith(b"RIFF"))
        await tts.stop()

    async def test_synthesize_stream_yields_pcm_chunks(self):
        tts = await self._start_tts()

        chunks = [chunk async for chunk in tts.synthesize_stream("hello world")]

        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(chunk, bytes) for chunk in chunks))
        await tts.stop()

    async def test_speak_with_speed_change_uses_multi_chunk_stretching_without_crashing(
        self,
    ):
        pcm_chunk = b"\x00\x10" * 4_800

        async def pcm_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
            del text, voice_kind, voice_reference
            return pcm_chunk

        tts = await self._start_tts(synth=pcm_synth)
        text = " ".join(f"word{index}" for index in range(60))
        session = await tts.speak(text, speed=2.0)
        pcm = await session.collect()
        self.assertTrue(pcm)
        self.assertEqual(len(pcm) % 2, 0)
        await tts.stop()

    async def test_speak_inserts_punctuation_pause_between_segments_without_trailing_silence(
        self,
    ):
        frame_samples = round(PCM_SAMPLE_RATE * 5 / 1000.0)
        first_pcm = struct.pack(
            f"<{frame_samples * 6}h",
            *([4_000] * (frame_samples * 6)),
        )
        second_pcm = struct.pack(
            f"<{frame_samples * 9}h",
            *(
                [12_000] * frame_samples
                + [0] * (frame_samples * 2)
                + [4_000] * (frame_samples * 6)
            ),
        )

        async def text_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
            del voice_kind, voice_reference
            if text == "  alpha.":
                return first_pcm
            self.assertEqual(text, "  beta.")
            return second_pcm

        tts = await self._start_tts(synth=text_synth)
        with patch("trillim.components.tts.public.random.uniform", return_value=1.0):
            pcm = await (await tts.speak("alpha. beta.")).collect()
            expected_pause = _pcm_silence(_boundary_pause_ms("  alpha."))
            expected_pcm = (
                _postprocess_segment_pcm(first_pcm, text="  alpha.", speed=1.0, add_pause=True)
                + _postprocess_segment_pcm(second_pcm, text="  beta.", speed=1.0, add_pause=False)
            )

        self.assertEqual(pcm, expected_pcm)
        self.assertFalse(pcm.endswith(expected_pause))
        await tts.stop()

    def test_segment_pause_pcm_scales_jittered_pause_by_speed(self):
        with patch("trillim.components.tts.public.random.uniform", return_value=1.1):
            pause = _segment_pause_pcm("alpha.", 2.0)

        expected_duration_ms = _boundary_pause_ms("alpha.") * 1.1 / 2.0
        expected_samples = round(PCM_SAMPLE_RATE * expected_duration_ms / 1000.0)
        expected_length = expected_samples * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
        self.assertEqual(len(pause), expected_length)

    def test_boundary_pause_helpers_cover_clause_and_non_boundary_cases(self):
        self.assertEqual(_boundary_pause_ms("alpha,"), 200)
        self.assertEqual(_boundary_pause_ms("alpha"), 0)
        self.assertEqual(_boundary_pause_ms("   "), 0)
        self.assertEqual(_segment_pause_pcm("alpha", 1.0), b"")
        self.assertEqual(_pcm_silence(0), b"")

    def test_apply_exponential_fade_in_pcm_ramps_start_and_preserves_tail(self):
        fade_frames = round(PCM_SAMPLE_RATE * 10 / 1000.0)
        pcm = struct.pack(
            f"<{fade_frames * PCM_CHANNELS}h",
            *([4_000] * (fade_frames * PCM_CHANNELS)),
        )

        faded = _apply_exponential_fade_in_pcm(pcm)
        faded_samples = memoryview(faded).cast("h")

        self.assertEqual(faded_samples[0], 0)
        self.assertLess(faded_samples[PCM_CHANNELS], 500)
        self.assertGreater(faded_samples[(fade_frames - 1) * PCM_CHANNELS], 3_900)
        self.assertEqual(_apply_exponential_fade_in_pcm(b""), b"")
        self.assertEqual(_apply_exponential_fade_in_pcm(b"\x00"), b"\x00")

    def test_postprocess_segment_pcm_composes_fade_trim_and_optional_pause(self):
        frame_samples = round(PCM_SAMPLE_RATE * 5 / 1000.0)
        pcm = struct.pack(
            f"<{frame_samples * 9}h",
            *(
                [12_000] * frame_samples
                + [0] * (frame_samples * 2)
                + [4_000] * (frame_samples * 6)
            ),
        )

        with patch("trillim.components.tts.public.random.uniform", return_value=1.0):
            self.assertEqual(
                _postprocess_segment_pcm(pcm, text="  alpha.", speed=1.0, add_pause=True),
                _apply_exponential_fade_in_pcm(pcm) + _pcm_silence(_boundary_pause_ms("  alpha.")),
            )
            self.assertEqual(
                _postprocess_segment_pcm(pcm, text="  alpha.", speed=1.0, add_pause=False),
                _apply_exponential_fade_in_pcm(pcm),
            )

    async def test_second_request_is_rejected_while_running(self):
        started = asyncio.Event()
        unblock = asyncio.Event()

        async def blocking_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
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
        with self.assertRaisesRegex(
            AdmissionRejectedError, "reservation is no longer active"
        ):
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
            with self.assertRaisesRegex(
                AdmissionRejectedError, "reservation is no longer active"
            ):
                await tts._start_reserved_session(
                    reservation,
                    "hello world",
                    voice="custom",
                    speed=1.0,
                )
        self.assertFalse(cleanup_path.exists())
        await tts.stop()

    async def test_failed_reserved_start_does_not_mask_admission_error_with_worker_close_failure(
        self,
    ):
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

        class _FailingWorker:
            async def close(self) -> None:
                raise RuntimeError("close boom")

        tts._session_worker_factory = lambda **_kwargs: _FailingWorker()
        with patch.object(
            tts._voice_store,
            "resolve_for_session",
            side_effect=resolve_for_session,
        ):
            with self.assertRaisesRegex(
                AdmissionRejectedError,
                "reservation is no longer active",
            ):
                await tts._start_reserved_session(
                    reservation,
                    "hello world",
                    voice="custom",
                    speed=1.0,
                )
        self.assertFalse(cleanup_path.exists())
        await tts.stop()

    async def test_failed_reserved_start_does_not_mask_admission_error_with_voice_cleanup_failure(
        self,
    ):
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

        original_unlink = Path.unlink

        def fail_voice_cleanup(path: Path, *args, **kwargs):
            if path == cleanup_path:
                raise PermissionError("cleanup boom")
            return original_unlink(path, *args, **kwargs)

        with (
            patch.object(
                tts._voice_store,
                "resolve_for_session",
                side_effect=resolve_for_session,
            ),
            patch(
                "pathlib.Path.unlink",
                autospec=True,
                side_effect=fail_voice_cleanup,
            ),
        ):
            with self.assertRaisesRegex(
                AdmissionRejectedError,
                "reservation is no longer active",
            ):
                await tts._start_reserved_session(
                    reservation,
                    "hello world",
                    voice="custom",
                    speed=1.0,
                )
        self.assertTrue(cleanup_path.exists())
        await tts.stop()

    async def test_stop_clears_reserved_slot(self):
        tts = await self._start_tts()
        reservation = await tts._reserve_session_slot()
        self.assertIs(tts._reserved_slot, reservation)
        await tts.stop()
        self.assertIsNone(tts._reserved_slot)

    async def test_release_reserved_slot_ignores_stale_tokens(self):
        tts = await self._start_tts()
        reservation = await tts._reserve_session_slot()

        await tts._release_reserved_slot(object())

        self.assertIs(tts._reserved_slot, reservation)
        await tts._release_reserved_slot(reservation)
        self.assertIsNone(tts._reserved_slot)
        await tts.stop()

    async def test_internal_get_tokenizer_loads_once_for_concurrent_callers(self):
        calls = 0
        tokenizer = object()

        def load_tokenizer():
            nonlocal calls
            calls += 1
            return tokenizer

        tts = TTS(_tokenizer_loader=load_tokenizer)

        first, second = await asyncio.gather(tts._get_tokenizer(), tts._get_tokenizer())

        self.assertIs(first, tokenizer)
        self.assertIs(second, tokenizer)
        self.assertEqual(calls, 1)
        self.assertIs(await tts._get_tokenizer(), tokenizer)

    async def test_scheduler_helper_noops_and_cancellation_paths(self):
        tts = await self._start_tts()
        session = self._make_internal_session(tts)

        await tts._pause_session(session)
        self.assertEqual(session.state, "running")

        tts._active_session = session
        await tts._pause_session(session)
        self.assertEqual(session.state, "paused")
        await tts._pause_session(session)
        self.assertEqual(session.state, "paused")
        await tts._resume_session(session)
        self.assertEqual(session.state, "running")

        waiter = asyncio.create_task(asyncio.sleep(0))
        session._task = waiter
        await tts._cancel_session(session)
        self.assertEqual(session.state, "cancelled")
        self.assertEqual(session.speed, 1.0)
        await tts.stop()

    async def test_wait_for_turn_and_start_session_helpers_cover_internal_branches(
        self,
    ):
        tts = await self._start_tts()
        session = self._make_internal_session(tts)
        existing_task = asyncio.create_task(asyncio.sleep(0))
        session._task = existing_task
        tts._start_session_locked(session)
        self.assertIs(session._task, existing_task)

        tts._active_session = None
        tts._pause_active_locked(session)
        self.assertEqual(session.state, "running")

        tts._active_session = session
        session._done_event.set()
        session._resume_event.set()
        with self.assertRaises(OperationCancelledError):
            await tts._wait_for_turn(session)

        session = self._make_internal_session(tts)
        tts._active_session = session
        session._pause_requested = True
        session._resume_event.set()

        async def release_pause() -> None:
            await asyncio.sleep(0)
            session._pause_requested = False
            session._set_running()

        releaser = asyncio.create_task(release_pause())
        self.assertEqual(await tts._wait_for_turn(session), 1.0)
        await releaser
        await tts.stop()

    async def test_check_can_accept_and_require_started_fail_closed(self):
        tts = TTS()

        with self.assertRaisesRegex(RuntimeError, "not started"):
            tts._require_started()

        tts._accepting = False
        with self.assertRaisesRegex(AdmissionRejectedError, "draining"):
            tts._check_can_accept_locked()

        tts._accepting = True
        tts._reserved_slot = object()
        with self.assertRaisesRegex(AdmissionRejectedError, "only one live session"):
            tts._check_can_accept_locked()

    async def test_resume_session_and_set_speed_cover_remaining_scheduler_branches(
        self,
    ):
        tts = await self._start_tts()
        session = self._make_internal_session(tts)

        await tts._resume_session(session)
        self.assertEqual(session.state, "running")

        tts._active_session = session
        await tts._set_session_speed(session, 1.5)
        self.assertEqual(session.speed, 1.5)

        await tts._resume_session(session)
        self.assertEqual(session.state, "running")
        await tts.stop()

    async def test_cancel_session_without_task_waits_for_explicit_finish(self):
        tts = await self._start_tts()
        session = self._make_internal_session(tts)
        tts._active_session = session

        async def finish_soon() -> None:
            await asyncio.sleep(0)
            await session._finish("cancelled", session._cancel_error())

        finisher = asyncio.create_task(finish_soon())
        await tts._cancel_session(session)
        await finisher

        self.assertEqual(session.state, "cancelled")
        await tts.stop()

    async def test_cancel_session_waits_for_inactive_session_cleanup(self):
        tts = await self._start_tts()
        session = self._make_internal_session(tts)

        async def finish_soon() -> None:
            await asyncio.sleep(0)
            await session._finish("cancelled", session._cancel_error())

        finisher = asyncio.create_task(finish_soon())
        await tts._cancel_session(session)
        await finisher

        self.assertEqual(session.state, "cancelled")
        await tts.stop()

    async def test_wait_for_turn_handles_inactive_session_before_owner_switches(self):
        tts = await self._start_tts()
        session = self._make_internal_session(tts)
        tts._active_session = self._make_internal_session(tts)
        session._resume_event.set()

        class _YieldingLock:
            def __init__(self) -> None:
                self.calls = 0

            async def __aenter__(self):
                self.calls += 1
                if self.calls == 2:
                    tts._active_session = session
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb

        tts._scheduler_lock = _YieldingLock()
        self.assertEqual(await tts._wait_for_turn(session), 1.0)
        await tts.stop()

    async def test_get_tokenizer_double_checked_lock_and_streaming_stretcher_edges(
        self,
    ):
        calls = 0
        tokenizer = object()

        def load_tokenizer():
            nonlocal calls
            calls += 1
            return tokenizer

        class _GateLock:
            def __init__(self) -> None:
                self._lock = asyncio.Lock()
                self.entered = asyncio.Event()
                self.release = asyncio.Event()
                self._first = True

            async def __aenter__(self):
                await self._lock.acquire()
                if self._first:
                    self._first = False
                    self.entered.set()
                    await self.release.wait()
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb
                self._lock.release()

        tts = TTS(_tokenizer_loader=load_tokenizer)
        tts._tokenizer_lock = _GateLock()
        first = asyncio.create_task(tts._get_tokenizer())
        await tts._tokenizer_lock.entered.wait()
        second = asyncio.create_task(tts._get_tokenizer())
        tts._tokenizer_lock.release.set()
        self.assertEqual(await first, tokenizer)
        self.assertEqual(await second, tokenizer)
        self.assertEqual(calls, 1)

        import numpy as np

        with patch("numpy.hanning", return_value=np.zeros(1024, dtype=np.float32)):
            zero_window = _StreamingPCMStretcher(1.25)
        self.assertTrue(np.all(zero_window._window == 1.0))

        stretcher = _StreamingPCMStretcher(1.25)
        self.assertEqual(stretcher.finish(), b"")
        self.assertEqual(stretcher.push(b"\x00"), b"")
        self.assertEqual(stretcher._pending_byte, b"\x00")
        self.assertEqual(stretcher.push(b"\x00\x00"), b"")
        self.assertEqual(stretcher.push(b"\x00\x00"), b"")
        self.assertIsNone(stretcher._get_spectrum(999))
        stretcher._process_output_frames(final=False)
        stretcher._processed_output_frames = 1
        self.assertEqual(stretcher._emit_ready(final=False), b"")
        stretcher._drop_consumed_spectra(2)

        np_module = stretcher._np
        stretcher._output = np_module.zeros(
            stretcher.frame_size * 2, dtype=np_module.float32
        )
        stretcher._weights = np_module.zeros(
            stretcher.frame_size * 2, dtype=np_module.float32
        )
        stretcher._processed_output_frames = 0
        stretcher._add_output_frame(
            np_module.zeros(stretcher.frame_size, dtype=np_module.float32)
        )

        stretcher = _StreamingPCMStretcher(1.5)
        stretcher.push((b"\x00\x00") * 10)
        self.assertIsInstance(stretcher.finish(), bytes)

        broken = _StreamingPCMStretcher(1.5)
        fake_spectrum = broken._np.zeros(
            broken.frame_size // 2 + 1, dtype=broken._np.complex64
        )
        broken._spectra.append(fake_spectrum)
        broken._spectra_base = 1
        broken._phase = broken._np.zeros(
            broken.frame_size // 2 + 1,
            dtype=broken._np.float32,
        )
        broken._next_time_step = 0.25
        broken._process_output_frames(final=True)

        class _ReportedShortInput:
            def __init__(
                self, np_module, *, reported_size: int, actual_size: int
            ) -> None:
                self._np = np_module
                self.size = reported_size
                self._actual = np_module.zeros(actual_size, dtype=np_module.float32)

            def __getitem__(self, item):
                if (
                    isinstance(item, slice)
                    and item.stop is None
                    and item.start == forced.hop_size
                ):
                    return self._actual.copy()
                return self._actual[item]

        forced = _StreamingPCMStretcher(1.25)
        forced._input = _ReportedShortInput(
            forced._np,
            reported_size=forced.frame_size,
            actual_size=forced.frame_size,
        )
        forced._total_samples = forced.frame_size
        forced._next_analysis_frame = 1
        forced._materialize_analysis_frames(final=True)
        self.assertEqual(len(forced._spectra), 1)

    async def test_second_request_is_rejected_while_paused(self):
        first_segment_started = asyncio.Event()
        first_segment_release = asyncio.Event()
        synth_calls = 0

        async def gated_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
            del text, voice_kind, voice_reference
            nonlocal synth_calls
            synth_calls += 1
            if synth_calls == 1:
                first_segment_started.set()
                await first_segment_release.wait()
            return b"pcm"

        tts = await self._start_tts(synth=gated_synth)
        text = " ".join(f"word{index}" for index in range(TARGET_TTS_TOKENS + 5))
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

        async def blocking_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
            del voice_kind, voice_reference
            seen_texts.append(text)
            if len(seen_texts) == 1:
                first_segment_started.set()
                await first_segment_release.wait()
            return text.encode("utf-8")

        tts = await self._start_tts(synth=blocking_synth)
        first = await tts.speak(
            " ".join(f"word{index}" for index in range(TARGET_TTS_TOKENS + 2))
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

        async def hanging_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
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

        async def hanging_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
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

    async def test_completed_sessions_ignore_pause_resume_cancel_and_speed_updates(
        self,
    ):
        tts = await self._start_tts()
        session = await tts.speak("hello world")
        self.assertTrue(await asyncio.wait_for(session.collect(), timeout=1))
        self.assertEqual(session.state, "completed")
        original_speed = session.speed

        await session.pause()
        await session.resume()
        await session.cancel()
        await session.set_speed(1.5)

        self.assertEqual(session.state, "completed")
        self.assertEqual(session.speed, original_speed)
        await tts.stop()

    async def test_pause_then_cancel_under_backpressure_cancels_blocked_put(self):
        allow_progress = asyncio.Event()
        blocked_put = asyncio.Event()
        put_calls = 0

        async def fast_synth(
            text: str, *, voice_kind: str, voice_reference: str
        ) -> bytes:
            del text, voice_kind, voice_reference
            await allow_progress.wait()
            return b"pcm"

        tts = await self._start_tts(synth=fast_synth)
        text = " ".join(
            f"word{index}"
            for index in range((MAX_EMITTED_AUDIO_CHUNKS + 2) * TARGET_TTS_TOKENS)
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
            lambda: (
                session._audio_queue.full()
                and session._task is not None
                and not session._task.done()
            )
        )
        await session.pause()
        self.assertEqual(session.state, "running")
        await asyncio.wait_for(session.cancel(), timeout=1)
        self.assertIsNotNone(session._task)
        assert session._task is not None
        self.assertTrue(session._task.done())
        await tts.stop()

    async def test_tampered_custom_voice_store_fails_closed_but_builtin_tts_still_works(
        self,
    ):
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
        self.assertTrue(pcm)
        with self.assertRaisesRegex(Exception, "tampered"):
            await tts.list_voices()
        with self.assertRaisesRegex(Exception, "tampered"):
            await tts.speak("hello again", voice="custom")
        await tts.stop()
