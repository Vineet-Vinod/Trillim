# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for TTS engine, component lifecycle, and server routes."""

import asyncio
import builtins
from collections import deque
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import trillim.server._tts as tts_module
from trillim.server import TTS
from trillim.server._tts import (
    MAX_BUFFER,
    MIN_CHUNK,
    SentenceChunker,
    TTSEngine,
    TTSSession,
    _SESSION_END,
    _StreamingPCMStretcher,
    wav_header,
)


class _FakeTensor:
    def __init__(self, samples):
        self._samples = samples

    def numpy(self):
        import numpy as np

        return np.array(self._samples, dtype=np.float32)


class _FakePocketModel:
    def __init__(self, chunks=None):
        self.sample_rate = 22050
        self.eval_calls = 0
        self.state_calls: list[tuple[object, bool]] = []
        self._chunks = chunks or [[0.0, 0.5, -0.5]]

    def eval(self):
        self.eval_calls += 1

    def get_state_for_audio_prompt(self, prompt, truncate=False):
        self.state_calls.append((prompt, truncate))
        return {"prompt": prompt, "truncate": truncate}

    def generate_audio_stream(self, **_):
        for chunk in self._chunks:
            yield _FakeTensor(chunk)


class _ManagedComponentEngine:
    DEFAULT_VOICE = "alba"

    def __init__(self, *args, **kwargs):
        self.init = kwargs
        self.default_voice = kwargs["default_voice"]
        self.sample_rate = 24000
        self.speed = kwargs["speed"]
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self):
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1

    def list_voices(self):
        return [{"voice_id": "alba", "name": "alba", "type": "predefined"}]


class _LoopStub:
    def __init__(self):
        self.calls = []

    def call_soon_threadsafe(self, callback, *args):
        self.calls.append((callback, args))


class _IteratorStub:
    def __init__(self, values, *, error=None, aclose_error=None):
        self._values = deque(values)
        self._error = error
        self._aclose_error = aclose_error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._error is not None and not self._values:
            raise self._error
        if not self._values:
            raise StopAsyncIteration
        value = self._values.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    async def aclose(self):
        self.closed = True
        if self._aclose_error is not None:
            raise self._aclose_error


class _BlockingIterator:
    def __init__(self, *, aclose_error=None):
        self._aclose_error = aclose_error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(1)
        return b"blocked"

    async def aclose(self):
        self.closed = True
        if self._aclose_error is not None:
            raise self._aclose_error


class _SessionEngine:
    def __init__(self, iterator_factory):
        self.sample_rate = 24000
        self.speed = 1.0
        self.iterator_factory = iterator_factory

    def synthesize_stream(self, text, voice=None, speed=None):
        return self.iterator_factory(text, voice, speed)


class TTSEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_wav_header_and_stretcher_edge_paths(self):
        header = wav_header(sample_rate=16000, bits_per_sample=16, channels=2, data_size=100)
        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(header[8:12], b"WAVE")
        self.assertEqual(int.from_bytes(header[24:28], "little"), 16000)
        self.assertEqual(int.from_bytes(header[40:44], "little"), 100)

        import numpy as np

        with patch("numpy.hanning", side_effect=lambda size: np.zeros(size, dtype=np.float32)):
            stretcher = _StreamingPCMStretcher(1.0)
        self.assertTrue((stretcher._window == 1.0).all())

        with self.assertRaisesRegex(RuntimeError, "out of order"):
            stretcher._output_base = 5
            stretcher._add_output_frame(np.zeros(stretcher.frame_size, dtype=np.float32))

        stretcher = _StreamingPCMStretcher(1.0)
        stretcher._spectra_base = 1
        stretcher._spectra.clear()
        stretcher._phase = np.zeros(stretcher.frame_size // 2 + 1, dtype=np.float32)
        stretcher._next_time_step = 0.0
        stretcher._process_output_frames(final=False)
        self.assertEqual(stretcher._processed_output_frames, 0)

        stretcher = _StreamingPCMStretcher(1.0)
        stretcher._phase = np.zeros(stretcher.frame_size // 2 + 1, dtype=np.float32)
        stretcher._next_time_step = 0.0
        stretcher._analysis_frame_count = lambda: 2
        stretcher._get_spectrum = lambda frame_index: None
        stretcher._process_output_frames(final=False)
        self.assertEqual(stretcher._processed_output_frames, 0)

    async def test_engine_start_stop_load_voice_management_and_synthesis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir)
            (voices_dir / "custom.wav").write_bytes(b"wav")
            model = _FakePocketModel()
            engine = TTSEngine(voices_dir=voices_dir, default_voice="alba", speed=1.5)

            with patch.object(engine, "_load", return_value=model):
                await engine.start()

            self.assertEqual(engine.sample_rate, 22050)
            self.assertIn("custom", engine._custom_voice_files)
            self.assertEqual(engine._voice_states["alba"]["prompt"], "alba")
            self.assertEqual(model.state_calls[0], ("alba", False))

            self.assertEqual(engine.list_voices()[0]["type"], "predefined")
            self.assertIn(
                {"voice_id": "custom", "name": "custom", "type": "custom"},
                engine.list_voices(),
            )

            self.assertEqual(engine._get_voice_state("alba")["prompt"], "alba")
            self.assertEqual(engine._get_voice_state("custom")["truncate"], True)
            with self.assertRaisesRegex(ValueError, "Unknown voice"):
                engine._get_voice_state("missing")

            await engine.register_voice("fresh", b"fresh-bytes")
            self.assertTrue((voices_dir / "fresh.wav").exists())
            self.assertIn("fresh", engine._voice_states)

            await engine.delete_voice("fresh")
            self.assertNotIn("fresh", engine._custom_voice_files)

            engine.default_voice = "custom"
            await engine.delete_voice("custom")
            self.assertEqual(engine.default_voice, engine.DEFAULT_VOICE)

            chunks = [chunk async for chunk in engine.synthesize_stream("hello", voice="alba", speed=1.0)]
            wav_bytes = await engine.synthesize_full("hello", voice="alba", speed=1.0)

            self.assertTrue(chunks)
            self.assertTrue(wav_bytes.startswith(b"RIFF"))

            await engine.stop()
            self.assertIsNone(engine._model)
            self.assertEqual(engine._voice_states, {})
            self.assertEqual(engine._custom_voice_files, {})

    async def test_engine_load_and_voice_errors(self):
        engine = TTSEngine(voices_dir=None)

        with self.assertRaisesRegex(RuntimeError, "TTSEngine not started"):
            [chunk async for chunk in engine.synthesize_stream("hello")]

        with self.assertRaisesRegex(RuntimeError, "TTSEngine not started"):
            await engine.register_voice("voice", b"bytes")

        engine._model = _FakePocketModel()
        with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
            await engine.register_voice("alba", b"bytes")

        with self.assertRaisesRegex(RuntimeError, "No voices directory configured"):
            await engine.register_voice("voice", b"bytes")

        engine = TTSEngine(voices_dir=Path(tempfile.mkdtemp()))
        engine._model = _FakePocketModel()
        with self.assertRaisesRegex(ValueError, "Invalid voice_id"):
            await engine.register_voice("../bad", b"bytes")

        with self.assertRaisesRegex(ValueError, "cannot be deleted"):
            await engine.delete_voice("alba")
        with self.assertRaisesRegex(KeyError, "not found"):
            await engine.delete_voice("missing")

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pocket_tts":
                raise ModuleNotFoundError("missing pocket_tts")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("trillim.server._tts.Path.exists", side_effect=[False, True]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Voice Optional Dependencies"):
                TTSEngine()._load()

        captured = {}
        fake_module = ModuleType("pocket_tts")

        class FakeTTSModel:
            @staticmethod
            def load_model():
                captured["loaded"] = True
                return _FakePocketModel()

        fake_module.TTSModel = FakeTTSModel
        with patch.dict(sys.modules, {"pocket_tts": fake_module}):
            model = TTSEngine()._load()
        self.assertTrue(captured["loaded"])
        self.assertIsInstance(model, _FakePocketModel)


class SentenceChunkerTests(unittest.TestCase):
    def test_sentence_chunker_force_flush_paths(self):
        chunker = SentenceChunker()
        sentence_text = ("a" * (MAX_BUFFER - 2)) + ". next"
        clause_text = ("b" * (MAX_BUFFER - 2)) + "; next"
        hard_text = "c" * (MAX_BUFFER + 1)

        parts = chunker.feed(sentence_text)
        self.assertEqual(parts, ["a" * (MAX_BUFFER - 2) + "."])
        self.assertEqual(chunker.flush(), "next")

        parts = chunker.feed(clause_text)
        self.assertEqual(parts, ["b" * (MAX_BUFFER - 2) + ";"])
        self.assertEqual(chunker.flush(), "next")

        parts = chunker.feed(hard_text)
        self.assertEqual(parts, [hard_text])
        self.assertIsNone(chunker.flush())


class TTSSessionEdgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_constructor_schedule_and_finish_edge_cases(self):
        tts = TTS()
        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)

        tts._loop = _LoopStub()
        tts._engine = SimpleNamespace(sample_rate=24000)
        session = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        self.assertIsNone(session.error)

        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            session.pause()
        self.assertEqual(tts._loop.calls[0][0].__name__, "_pause_session")

        session._finish("failed", ValueError("bad"))
        self.assertEqual(session.state, "failed")
        self.assertIsInstance(session.error, ValueError)
        session._set_state("running")
        session._finish("completed")
        self.assertEqual(session.state, "failed")

        with self.assertRaisesRegex(ValueError, "bad"):
            await session.collect()

    async def test_session_stop_alias_and_cancel_paths(self):
        loop = asyncio.get_running_loop()
        tts = TTS()
        tts._loop = loop
        tts._engine = SimpleNamespace(sample_rate=24000)
        tts._active_session = None
        tts._queued_sessions = deque()
        cancelled = []
        tts._cancel_session = lambda session: cancelled.append(session)

        session = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        session.stop()
        self.assertEqual(cancelled, [session])


class TTSComponentTests(unittest.IsolatedAsyncioTestCase):
    async def test_component_start_timeout_validation_and_queue_edge_cases(self):
        loop = asyncio.get_running_loop()
        engine = _ManagedComponentEngine(voices_dir=Path(tempfile.mkdtemp()), default_voice="alba", speed=1.0)
        tts = TTS(speed=1.25)

        with patch("trillim.server._tts.TTSEngine", return_value=engine) as engine_cls:
            await tts.start()

        self.assertIs(tts.engine, engine)
        self.assertIs(tts._loop, loop)
        self.assertEqual(engine.start_calls, 1)
        engine_cls.assert_called_once()

        with self.assertRaisesRegex(ValueError, "timeout must be > 0"):
            tts._validate_timeout(0)

        tts._loop = None
        session = object.__new__(TTSSession)
        with self.assertRaisesRegex(RuntimeError, "TTS not started"):
            TTS()._start_session(session)

        tts._loop = loop
        active = object()
        queued_done = SimpleNamespace(done=True)
        next_session = SimpleNamespace(done=False)
        tts._active_session = active
        tts._queued_sessions = deque([queued_done, next_session])
        tts._start_next_session()
        self.assertIs(tts._active_session, active)

        tts._active_session = None
        started = []
        tts._start_session = lambda session: started.append(session)
        tts._start_next_session()
        self.assertEqual(started, [next_session])

        await tts.stop()
        self.assertIsNone(tts.engine)

    async def test_pause_resume_cancel_and_drain_internal_paths(self):
        tts = TTS()
        tts._loop = asyncio.get_running_loop()
        tts._engine = SimpleNamespace(sample_rate=24000)
        tts._queued_sessions = deque()
        tts._active_session = None

        session = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        session._finish("completed")
        tts._pause_session(session)
        tts._resume_session(session)
        tts._cancel_session(session)
        self.assertEqual(session.state, "completed")

        session = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        tts._queued_sessions = deque([session])
        tts._resume_session(session)
        self.assertEqual(session.state, "queued")
        tts._queued_sessions.clear()
        tts._cancel_session(session)
        self.assertEqual(session.state, "cancelled")

        active = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        tts._active_session = active
        tts._cancel_session(active)
        self.assertEqual(active.state, "cancelled")

        iterator = _IteratorStub([b"a"])
        cancelled = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        cancelled._state = "cancelled"
        await tts._drain_session(cancelled, iterator)

        iterator = _IteratorStub([b"a"])
        session = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        original_wait = session._resume_event.wait
        call_count = {"value": 0}

        async def wait_then_cancel():
            call_count["value"] += 1
            await original_wait()
            if call_count["value"] == 2:
                session._state = "cancelled"

        session._resume_event.wait = wait_then_cancel
        await tts._drain_session(session, iterator)
        self.assertEqual(session._chunks.qsize(), 0)

    async def test_run_session_error_cancel_and_aclose_paths(self):
        tts = TTS()
        tts._loop = asyncio.get_running_loop()
        tts._engine = SimpleNamespace(sample_rate=24000)
        tts._queued_sessions = deque()
        tts._active_session = None

        async def error_stream(text, voice, speed):
            raise ValueError("boom")
            yield b"never"

        tts._engine = _SessionEngine(lambda text, voice, speed: error_stream(text, voice, speed))
        session = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        await tts._run_session(session)
        self.assertEqual(session.state, "failed")
        self.assertIsInstance(session.error, ValueError)

        iterator = _BlockingIterator(aclose_error=RuntimeError("close failed"))
        tts._engine = _SessionEngine(lambda text, voice, speed: iterator)
        tts._active_session = None
        session = TTSSession(tts, text="hi", voice=None, speed=1.0, timeout=None)
        task = asyncio.create_task(tts._run_session(session))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(session.state, "cancelled")
        self.assertTrue(iterator.closed)


class TTSRouterTests(unittest.TestCase):
    def _make_app(self, tts: TTS) -> FastAPI:
        app = FastAPI()
        app.include_router(tts.router())
        return app

    def test_router_requires_python_multipart_dependency(self):
        tts = TTS()
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "python_multipart":
                raise ModuleNotFoundError("missing python_multipart")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("trillim.server._tts.Path.exists", side_effect=[False, True]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Voice Optional Dependencies"):
                tts.router()

    def test_routes_cover_status_errors_and_success(self):
        tts = TTS()
        tts._engine = _ManagedComponentEngine(voices_dir=Path(tempfile.mkdtemp()), default_voice="alba", speed=1.0)
        tts._loop = _LoopStub()
        tts.list_voices = lambda: [{"voice_id": "alba", "name": "alba", "type": "predefined"}]
        tts.register_voice = AsyncMock(return_value=None)
        tts.delete_voice = AsyncMock(return_value=None)

        async def synthesize_stream(*args, **kwargs):
            yield b"pcm"

        tts.synthesize_stream = synthesize_stream

        with TestClient(self._make_app(tts)) as client:
            tts._engine = None
            self.assertEqual(client.get("/v1/voices").status_code, 503)
            self.assertEqual(
                client.post("/v1/voices", files={"file": ("voice.wav", b"a", "audio/wav")}, data={"voice_id": "x"}).status_code,
                503,
            )
            self.assertEqual(client.delete("/v1/voices/x").status_code, 503)
            self.assertEqual(
                client.post("/v1/audio/speech", json={"input": "hi"}).status_code,
                503,
            )

            tts._engine = _ManagedComponentEngine(voices_dir=Path(tempfile.mkdtemp()), default_voice="alba", speed=1.0)
            response = client.get("/v1/voices")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["voices"][0]["voice_id"], "alba")

            response = client.post(
                "/v1/voices",
                files={"file": ("voice.wav", b"a", "audio/wav")},
                data={"voice_id": "new"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"voice_id": "new", "status": "created"})

            tts.register_voice = AsyncMock(side_effect=ValueError("bad voice"))
            response = client.post(
                "/v1/voices",
                files={"file": ("voice.wav", b"a", "audio/wav")},
                data={"voice_id": "bad"},
            )
            self.assertEqual(response.status_code, 400)

            response = client.post(
                "/v1/voices",
                files={"file": ("voice.wav", b"x" * (8 * 1024 * 1024 + 1), "audio/wav")},
                data={"voice_id": "big"},
            )
            self.assertEqual(response.status_code, 413)

            response = client.delete("/v1/voices/new")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "deleted", "voice_id": "new"})

            tts.delete_voice = AsyncMock(side_effect=ValueError("bad delete"))
            self.assertEqual(client.delete("/v1/voices/bad").status_code, 400)
            tts.delete_voice = AsyncMock(side_effect=KeyError("missing"))
            self.assertEqual(client.delete("/v1/voices/missing").status_code, 404)

            self.assertEqual(
                client.post("/v1/audio/speech", json={"input": "   "}).status_code,
                400,
            )
            self.assertEqual(
                client.post("/v1/audio/speech", json={"input": "hi", "speed": 5.0}).status_code,
                400,
            )

            response = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "response_format": "pcm", "speed": 1.5},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"pcm")
            self.assertEqual(response.headers["content-type"], "audio/pcm")

            response = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "voice": "alba", "speed": 1.0},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.content.startswith(b"RIFF"))


if __name__ == "__main__":
    unittest.main()
