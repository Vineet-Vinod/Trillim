# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the public Whisper SDK helpers and component routes."""

import asyncio
import builtins
import io
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient

import trillim.server._whisper as whisper_module
from trillim.server import Whisper


class _FakeWhisperEngine:
    def __init__(self):
        self.calls: list[tuple[bytes, str | None]] = []

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        self.calls.append((audio_bytes, language))
        return "decoded"


class _SlowWhisperEngine(_FakeWhisperEngine):
    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        await asyncio.sleep(0.05)
        return await super().transcribe(audio_bytes, language=language)


class _ManagedWhisperEngine(_FakeWhisperEngine):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.init = kwargs
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self):
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1


class _ArrayLike:
    def __init__(self, values, *, kind=None, itemsize=None):
        self._values = values
        self.dtype = None if kind is None else SimpleNamespace(kind=kind, itemsize=itemsize)

    def tolist(self):
        return self._values


class _FakeSegment:
    def __init__(self, text: str):
        self.text = text


class _FakeModel:
    def __init__(self):
        self.calls: list[tuple[bytes, str | None, int]] = []

    def transcribe(self, audio_file, language=None, beam_size=5):
        self.calls.append((audio_file.read(), language, beam_size))
        return ([_FakeSegment(" hello "), _FakeSegment("world ")], None)


class WhisperSdkTests(unittest.IsolatedAsyncioTestCase):
    def _make_whisper(self, engine) -> Whisper:
        whisper = Whisper()
        whisper._engine = engine
        return whisper

    async def test_component_lifecycle_property_and_require_started(self):
        whisper = Whisper(model_size="small", compute_type="float16", cpu_threads=7)
        engine = _ManagedWhisperEngine()

        with self.assertRaisesRegex(RuntimeError, "Whisper not started"):
            whisper._require_started()

        with patch("trillim.server._whisper.WhisperEngine", return_value=engine) as engine_cls:
            await whisper.start()

        self.assertIs(whisper.engine, engine)
        self.assertEqual(engine.start_calls, 1)
        engine_cls.assert_called_once_with(
            model_size="small",
            compute_type="float16",
            cpu_threads=7,
        )

        await whisper.stop()

        self.assertEqual(engine.stop_calls, 1)

    async def test_transcribe_bytes_uses_active_engine(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        result = await whisper.transcribe_bytes(b"audio", language="en")

        self.assertEqual(result, "decoded")
        self.assertEqual(engine.calls, [(b"audio", "en")])

    async def test_transcribe_wav_reads_file_bytes(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(b"RIFFdemo")
            handle.flush()

            result = await whisper.transcribe_wav(handle.name, language="fr")

        self.assertEqual(result, "decoded")
        self.assertEqual(engine.calls, [(b"RIFFdemo", "fr")])

    async def test_transcribe_array_encodes_valid_wav_from_frames_first_audio(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        result = await whisper.transcribe_array(
            [[0.25, -0.25], [0.5, -0.5], [0.0, 0.0]],
            sample_rate=44100,
            language="en",
        )

        self.assertEqual(result, "decoded")
        wav_bytes, language = engine.calls[-1]
        self.assertEqual(language, "en")
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 44100)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getnframes(), 3)

    async def test_transcribe_array_accepts_channels_first_layout(self):
        engine = _FakeWhisperEngine()
        whisper = self._make_whisper(engine)

        await whisper.transcribe_array(
            [[0.25, 0.5, 0.75], [-0.25, -0.5, -0.75]],
            sample_rate=22050,
        )

        wav_bytes, _ = engine.calls[-1]
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 22050)
            self.assertEqual(wav_file.getnframes(), 3)

    async def test_transcribe_array_rejects_invalid_input(self):
        whisper = self._make_whisper(_FakeWhisperEngine())

        with self.assertRaisesRegex(ValueError, "sample_rate must be >= 1"):
            await whisper.transcribe_array([0.1, 0.2], sample_rate=0)

        with self.assertRaisesRegex(ValueError, "samples must not be empty"):
            await whisper.transcribe_array([], sample_rate=16000)

        with self.assertRaisesRegex(TypeError, "array-like sequence"):
            await whisper.transcribe_array(b"raw-bytes", sample_rate=16000)

    async def test_transcribe_methods_support_timeout(self):
        whisper = self._make_whisper(_SlowWhisperEngine())

        with self.assertRaisesRegex(TimeoutError, "Whisper transcription timed out"):
            await whisper.transcribe_bytes(b"audio", timeout=0.001)


class WhisperEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_engine_start_stop_transcribe_and_sync_join(self):
        model = _FakeModel()
        engine = whisper_module.WhisperEngine()

        with patch.object(engine, "_load", return_value=model):
            await engine.start()

        self.assertIs(engine._model, model)
        self.assertEqual(
            await engine.transcribe(b"audio", language="fr"),
            "hello world",
        )
        self.assertEqual(model.calls, [(b"audio", "fr", 5)])

        await engine.stop()
        self.assertIsNone(engine._model)

    async def test_engine_transcribe_requires_start(self):
        engine = whisper_module.WhisperEngine()

        with self.assertRaisesRegex(RuntimeError, "WhisperEngine not started"):
            await engine.transcribe(b"audio")

    def test_engine_load_formats_import_errors_and_success(self):
        engine = whisper_module.WhisperEngine(model_size="tiny", compute_type="float32", cpu_threads=9)
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "faster_whisper":
                raise ModuleNotFoundError("missing whisper")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("trillim.server._whisper.Path.exists", side_effect=[False, True]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Voice Optional Dependencies"):
                engine._load()

        captured = {}
        fake_module = ModuleType("faster_whisper")

        class FakeWhisperModel:
            def __init__(self, model_size, *, device, compute_type, cpu_threads):
                captured["args"] = (model_size, device, compute_type, cpu_threads)

        fake_module.WhisperModel = FakeWhisperModel
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            model = engine._load()

        self.assertIsInstance(model, FakeWhisperModel)
        self.assertEqual(captured["args"], ("tiny", "cpu", "float32", 9))


class WhisperRouterTests(unittest.TestCase):
    def _make_app(self, whisper: Whisper) -> FastAPI:
        app = FastAPI()
        app.include_router(whisper.router())
        return app

    def test_router_requires_python_multipart_dependency(self):
        whisper = Whisper()
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "python_multipart":
                raise ModuleNotFoundError("missing python_multipart")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch("trillim.server._whisper.Path.exists", side_effect=[False, True]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Voice Optional Dependencies"):
                whisper.router()

    def test_transcriptions_route_covers_status_json_text_and_upload_limit(self):
        whisper = Whisper()
        whisper._engine = _FakeWhisperEngine()
        whisper.transcribe_bytes = AsyncMock(return_value="decoded")

        with TestClient(self._make_app(whisper)) as client:
            whisper._engine = None
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", b"abc", "audio/wav")},
            )
            self.assertEqual(response.status_code, 503)

            whisper._engine = _FakeWhisperEngine()
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", b"abc", "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"text": "decoded"})
            whisper.transcribe_bytes.assert_awaited_with(b"abc", language=None)

            response = client.post(
                "/v1/audio/transcriptions",
                data={"language": "en", "response_format": "text"},
                files={"file": ("sample.wav", b"123", "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "decoded")
            whisper.transcribe_bytes.assert_awaited_with(b"123", language="en")

            response = client.post(
                "/v1/audio/transcriptions",
                files={
                    "file": (
                        "sample.wav",
                        b"x" * (8 * 1024 * 1024 + 1),
                        "audio/wav",
                    )
                },
            )
            self.assertEqual(response.status_code, 413)


class WhisperHelperEdgeCaseTests(unittest.TestCase):
    def test_array_helpers_cover_edge_paths(self):
        self.assertEqual(
            whisper_module._coerce_mono_samples(_ArrayLike([0.0, 0.5])),
            [0.0, 0.5],
        )
        self.assertEqual(
            whisper_module._coerce_mono_samples(iter([0.25, -0.25])),
            [0.25, -0.25],
        )

        with self.assertRaisesRegex(ValueError, "samples must not be empty"):
            whisper_module._collapse_channels([[]])
        with self.assertRaisesRegex(ValueError, "consistent shape"):
            whisper_module._collapse_channels([[1.0], [1.0, 2.0]])

        self.assertEqual(
            whisper_module._infer_scale_hint(_ArrayLike([1], kind="i", itemsize=2)),
            32768.0,
        )
        self.assertEqual(
            whisper_module._infer_scale_hint(_ArrayLike([1], kind="u", itemsize=1)),
            255.0,
        )

        with self.assertRaisesRegex(TypeError, "Unsupported sample value"):
            whisper_module._infer_scale_hint([True, object()])

        scale_cases = {
            (2.0,): 32768.0,
            (100000.0,): 8388608.0,
            (100000000.0,): 2147483648.0,
            (10000000000.0,): 10000000000.0,
        }
        for values, expected in scale_cases.items():
            with self.subTest(values=values):
                self.assertEqual(whisper_module._infer_scale_hint(list(values)), expected)

        self.assertEqual(
            list(whisper_module._flatten(_ArrayLike([1.0, [2.0, 3.0]]))),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(list(whisper_module._flatten(4.0)), [4.0])

        self.assertEqual(whisper_module._normalize_scalar(True, scale_hint=None), 1.0)
        self.assertEqual(whisper_module._normalize_scalar(10.0, scale_hint=2.0), 1.0)
        self.assertEqual(whisper_module._normalize_scalar(-10.0, scale_hint=2.0), -1.0)
        self.assertEqual(whisper_module._float_to_int16(2.0), 32767)
        self.assertEqual(whisper_module._float_to_int16(-2.0), -32768)
        self.assertTrue(whisper_module._is_sequence(_ArrayLike([1.0])))
        self.assertFalse(whisper_module._is_sequence(1.0))


if __name__ == "__main__":
    unittest.main()
