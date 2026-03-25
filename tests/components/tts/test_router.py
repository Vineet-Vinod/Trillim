"""Tests for the TTS HTTP router."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from trillim.components.tts._router import build_router
from trillim.components.tts._router import _as_http_error
from trillim.components.tts._validation import PayloadTooLargeError, validate_http_voice_upload_request
from trillim.components.tts._worker import WorkerFailureError
from trillim.errors import AdmissionRejectedError, InvalidRequestError, ProgressTimeoutError
from trillim.server import Server
from tests.components.tts.support import FakeRequest, make_started_tts


class TTSRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self.spool_dir = Path(self._temp_dir.name) / "spool"

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    def _make_server(self, **kwargs) -> tuple[Server, patch, patch, patch]:
        tts, imports_patch, builtins_patch = make_started_tts(**kwargs)
        tts._spool_dir = self.spool_dir
        return (
            Server(tts),
            patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root),
            imports_patch,
            builtins_patch,
        )

    async def test_duplicate_voice_is_rejected_before_body_consumption(self):
        tts, imports_patch, builtins_patch = make_started_tts()
        tts._spool_dir = self.spool_dir
        with patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root), builtins_patch, imports_patch:
            await tts.start()
        await tts.register_voice("custom", b"voice")
        request = FakeRequest(headers={"name": "custom"}, chunks=[b"voice"])
        request.state.trillim_tts_voice_request = validate_http_voice_upload_request(
            content_length="5",
            name="custom",
        )
        with self.assertRaisesRegex(InvalidRequestError, "already exists"):
            await tts._register_voice_http_request(request)
        self.assertFalse(request.stream_called)

    def test_voices_routes_and_speech_sse(self):
        server, root_patch, imports_patch, builtins_patch = self._make_server()
        with root_patch, builtins_patch, imports_patch:
            with TestClient(server.app) as client:
                create = client.post("/v1/voices", content=b"voice", headers={"name": "custom"})
                self.assertEqual(create.status_code, 200)
                self.assertEqual(create.json(), {"name": "custom", "status": "created"})
                self.assertEqual(client.get("/v1/voices").json(), {"voices": ["alba", "marius", "custom"]})
                with client.stream(
                    "POST",
                    "/v1/audio/speech",
                    content="hello world".encode("utf-8"),
                    headers={"voice": "alba", "speed": "1.0"},
                ) as response:
                    body = "".join(chunk.decode("utf-8") for chunk in response.iter_bytes())
                self.assertEqual(response.status_code, 200)
                self.assertIn("event: audio", body)
                self.assertIn("event: done", body)
                deleted = client.delete("/v1/voices/custom")
                self.assertEqual(deleted.json(), {"name": "custom", "status": "deleted"})

    def test_audio_speech_rejects_invalid_utf8(self):
        server, root_patch, imports_patch, builtins_patch = self._make_server()
        with root_patch, builtins_patch, imports_patch:
            with TestClient(server.app) as client:
                response = client.post(
                    "/v1/audio/speech",
                    content=b"\xff",
                    headers={"voice": "alba"},
                )
        self.assertEqual(response.status_code, 400)

    async def test_audio_speech_rejects_busy_request_before_reading_body(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            started.set()
            await release.wait()
            return b"pcm"

        tts, imports_patch, builtins_patch = make_started_tts(synth=blocking_synth)
        tts._spool_dir = self.spool_dir
        with patch("trillim.components.tts.public.VOICE_STORE_ROOT", self.voice_root), builtins_patch, imports_patch:
            await tts.start()
        session = await tts.speak("hello world")
        await started.wait()
        router = build_router(tts)
        endpoint = next(route.endpoint for route in router.routes if route.path == "/v1/audio/speech")
        request = FakeRequest(
            headers={"content-length": "5", "voice": "alba"},
            chunks=[b"hello"],
        )
        with patch(
            "trillim.components.tts._router._read_bounded_body",
            side_effect=AssertionError("body should not be read when TTS is busy"),
        ):
            with self.assertRaises(HTTPException) as context:
                await endpoint(request)
        self.assertEqual(context.exception.status_code, 429)
        self.assertFalse(request.stream_called)
        release.set()
        await session.cancel()
        await tts.stop()

    def test_voice_upload_maps_worker_failures_to_400(self):
        async def failing_builder(_audio_path: Path) -> bytes:
            raise WorkerFailureError("unsupported or malformed audio input")

        server, root_patch, imports_patch, builtins_patch = self._make_server(
            voice_state_builder=failing_builder
        )
        with root_patch, builtins_patch, imports_patch:
            with TestClient(server.app) as client:
                response = client.post(
                    "/v1/voices",
                    content=b"voice",
                    headers={"name": "custom"},
                )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "unsupported or malformed audio input")

    def test_voice_upload_maps_backend_worker_failures_to_503(self):
        async def failing_builder(_audio_path: Path) -> bytes:
            raise WorkerFailureError("backend voice builder crashed")

        server, root_patch, imports_patch, builtins_patch = self._make_server(
            voice_state_builder=failing_builder
        )
        with root_patch, builtins_patch, imports_patch:
            with TestClient(server.app) as client:
                response = client.post(
                    "/v1/voices",
                    content=b"voice",
                    headers={"name": "custom"},
                )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "backend voice builder crashed")

    def test_audio_speech_streams_error_events_and_http_error_mapping(self):
        async def failing_synth(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
            del text, voice_kind, voice_reference
            raise RuntimeError("boom")

        server, root_patch, imports_patch, builtins_patch = self._make_server(synth=failing_synth)
        with root_patch, builtins_patch, imports_patch:
            with TestClient(server.app) as client:
                with client.stream(
                    "POST",
                    "/v1/audio/speech",
                    content=b"hello world",
                    headers={"voice": "alba"},
                ) as response:
                    body = "".join(chunk.decode("utf-8") for chunk in response.iter_bytes())
                self.assertEqual(response.status_code, 200)
                self.assertIn("event: error", body)

        self.assertEqual(_as_http_error(PayloadTooLargeError("too big")).status_code, 413)
        self.assertEqual(_as_http_error(InvalidRequestError("bad")).status_code, 400)
        self.assertEqual(_as_http_error(AdmissionRejectedError("busy")).status_code, 429)
        self.assertEqual(_as_http_error(ProgressTimeoutError("slow")).status_code, 504)
