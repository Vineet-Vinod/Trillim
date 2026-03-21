"""Tests for the STT HTTP router."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from trillim.components.stt._router import _as_http_error, _handle_transcription_request
from trillim.components.stt._validation import PayloadTooLargeError
from trillim.components.stt.public import STT
from trillim.errors import AdmissionRejectedError, InvalidRequestError, ProgressTimeoutError
from trillim.server import Server
from tests.components.stt.support import FakeRequest, make_faster_whisper_stub


class STTRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.spool_dir = Path(self._temp_dir.name) / "spool"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    def _make_server(self) -> Server:
        stt = STT()
        stt._spool_dir = self.spool_dir
        return Server(stt)

    async def _make_started_stt(self) -> STT:
        stt = STT()
        stt._spool_dir = self.spool_dir
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}):
            await stt.start()
        return stt

    def test_audio_transcriptions_returns_final_json(self):
        server = self._make_server()

        async def fake_worker(audio_path, *, language):
            return "hello"

        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}), patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=fake_worker,
        ):
            with TestClient(server.app) as client:
                response = client.post(
                    "/v1/audio/transcriptions?language=en",
                    content=b"abc",
                    headers={"content-type": "audio/wav"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hello"})

    def test_audio_transcriptions_rejects_invalid_content_type(self):
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}):
            with TestClient(self._make_server().app) as client:
                response = client.post(
                    "/v1/audio/transcriptions",
                    content=b"abc",
                    headers={"content-type": "text/plain"},
                )
        self.assertEqual(response.status_code, 400)
        self.assertIn("content-type", response.json()["detail"])

    def test_audio_transcriptions_rejects_invalid_language(self):
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}):
            with TestClient(self._make_server().app) as client:
                response = client.post(
                    "/v1/audio/transcriptions?language=en_us",
                    content=b"abc",
                    headers={"content-type": "audio/wav"},
                )
        self.assertEqual(response.status_code, 400)
        self.assertIn("letters and hyphens", response.json()["detail"])

    def test_audio_transcriptions_rejects_oversize_upload(self):
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}), patch(
            "trillim.components.stt._validation.MAX_UPLOAD_BYTES",
            2,
        ):
            with TestClient(self._make_server().app) as client:
                response = client.post(
                    "/v1/audio/transcriptions",
                    content=b"abc",
                    headers={"content-type": "audio/wav"},
                )
        self.assertEqual(response.status_code, 413)

    async def test_busy_request_is_rejected_before_body_consumption(self):
        stt = await self._make_started_stt()
        lease = await stt._admission.acquire()
        request = FakeRequest(
            headers={"content-type": "audio/wav"},
            chunks=[b"abc"],
        )
        with self.assertRaisesRegex(AdmissionRejectedError, "STT is busy"):
            await _handle_transcription_request(stt, request)
        self.assertFalse(request.stream_called)
        await lease.release()

    async def test_content_length_over_limit_is_rejected_before_spooling(self):
        stt = await self._make_started_stt()
        request = FakeRequest(
            headers={"content-type": "audio/wav", "content-length": "4"},
            chunks=[b"abc"],
        )
        with patch("trillim.components.stt._validation.MAX_UPLOAD_BYTES", 3):
            with self.assertRaises(PayloadTooLargeError):
                await stt._transcribe_http_request(request)
        self.assertFalse(request.stream_called)

    async def test_shorter_than_claimed_body_is_accepted_if_non_empty(self):
        stt = await self._make_started_stt()

        async def fake_worker(audio_path, *, language):
            return "hello"

        request = FakeRequest(
            headers={"content-type": "audio/wav", "content-length": "10"},
            chunks=[b"abc"],
            query_params={"language": "en"},
        )
        with patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=fake_worker,
        ):
            text = await _handle_transcription_request(stt, request)
        self.assertEqual(text, "hello")
        self.assertTrue(request.stream_called)

    def test_audio_transcriptions_maps_worker_timeout_to_504(self):
        server = self._make_server()
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}), patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=ProgressTimeoutError("timed out"),
        ):
            with TestClient(server.app) as client:
                response = client.post(
                    "/v1/audio/transcriptions",
                    content=b"abc",
                    headers={"content-type": "audio/wav"},
                )
        self.assertEqual(response.status_code, 504)

    def test_audio_transcriptions_maps_worker_failure_to_503(self):
        server = self._make_server()
        with patch.dict("sys.modules", {"faster_whisper": make_faster_whisper_stub()}), patch(
            "trillim.components.stt.public.transcribe_owned_audio_file",
            side_effect=RuntimeError("boom"),
        ):
            with TestClient(server.app) as client:
                response = client.post(
                    "/v1/audio/transcriptions",
                    content=b"abc",
                    headers={"content-type": "audio/wav"},
                )
        self.assertEqual(response.status_code, 503)

    def test_server_startup_fails_when_dependency_is_missing(self):
        with patch(
            "trillim.components.stt.public.importlib.import_module",
            side_effect=ModuleNotFoundError("faster_whisper"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Component startup failed"):
                with TestClient(self._make_server().app):
                    pass

    def test_as_http_error_maps_known_cases(self):
        self.assertEqual(_as_http_error(PayloadTooLargeError("too big")).status_code, 413)
        self.assertEqual(_as_http_error(InvalidRequestError("bad")).status_code, 400)
        self.assertEqual(_as_http_error(AdmissionRejectedError("busy")).status_code, 429)
        self.assertEqual(_as_http_error(ProgressTimeoutError("slow")).status_code, 504)
