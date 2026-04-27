from __future__ import annotations

import base64
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

from trillim.components.tts import TTS
from trillim.components.tts._limits import MAX_VOICE_UPLOAD_BYTES
from trillim.components.tts._router import _as_http_error
from trillim.components.tts._validation import PayloadTooLargeError
from trillim.errors import (
    AdmissionRejectedError,
    ComponentLifecycleError,
    InvalidRequestError,
    ProgressTimeoutError,
)
from trillim.server import Server

from tests.components.tts.support import reference_wav_bytes, tts_voice_store_environment


class TTSRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self._stack = tts_voice_store_environment(self.voice_root)

    def tearDown(self) -> None:
        self._stack.close()
        self._temp_dir.cleanup()

    def _make_client(self) -> TestClient:
        return TestClient(Server(TTS()).app)

    def test_voice_routes_return_expected_json(self):
        with self._make_client() as client:
            voices = client.get("/v1/voices").json()["voices"]
            self.assertIn("alba", voices)
            self.assertNotIn("custom", voices)

            create = client.post(
                "/v1/voices",
                content=reference_wav_bytes(),
                headers={"name": "custom"},
            )
            self.assertEqual(create.status_code, 200)
            self.assertEqual(create.json(), {"name": "custom", "status": "created"})
            self.assertIn("custom", client.get("/v1/voices").json()["voices"])

            delete = client.delete("/v1/voices/custom")
            self.assertEqual(delete.status_code, 200)
            self.assertEqual(delete.json(), {"name": "custom", "status": "deleted"})

    def test_voice_routes_map_invalid_requests(self):
        with self._make_client() as client:
            cases = (
                ("post", "/v1/voices", reference_wav_bytes(), {}, 400, "name header is required"),
                ("post", "/v1/voices", b"", {"name": "custom"}, 400, "must not be empty"),
                (
                    "post",
                    "/v1/voices",
                    reference_wav_bytes(),
                    {"name": "bad-name"},
                    400,
                    "letters and digits",
                ),
                ("delete", "/v1/voices/alba", b"", {}, 400, "built in"),
                ("delete", "/v1/voices/missing", b"", {}, 404, "missing"),
            )
            for method, path, body, headers, status_code, detail in cases:
                with self.subTest(method=method, path=path):
                    if method == "delete":
                        response = client.delete(path, headers=headers)
                    else:
                        response = client.post(path, content=body, headers=headers)
                    self.assertEqual(response.status_code, status_code)
                    self.assertIn(detail, response.json()["detail"])

    def test_voice_upload_honors_content_length_limit_before_body_read(self):
        with self._make_client() as client:
            response = client.post(
                "/v1/voices",
                content=b"x",
                headers={
                    "name": "custom",
                    "content-length": str(MAX_VOICE_UPLOAD_BYTES + 1),
                },
            )

        self.assertEqual(response.status_code, 413)

    def test_audio_speech_streams_sse_audio_and_done(self):
        with self._make_client() as client:
            response = client.post("/v1/audio/speech", content=b"hello")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        body = response.text
        self.assertIn("event: audio\n", body)
        self.assertIn("event: done\n", body)
        self._assert_first_audio_event_is_pcm(body)

    def test_audio_speech_uses_registered_custom_voice_header(self):
        with self._make_client() as client:
            create = client.post(
                "/v1/voices",
                content=reference_wav_bytes(),
                headers={"name": "custom"},
            )
            self.assertEqual(create.status_code, 200)
            response = client.post(
                "/v1/audio/speech",
                content=b"hello",
                headers={"voice": "custom"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: audio\n", response.text)
        self._assert_first_audio_event_is_pcm(response.text)

    def test_audio_speech_maps_invalid_requests(self):
        with self._make_client() as client:
            cases = (
                ({}, b"", 400, "speech input must not be empty"),
                ({"voice": "bad-name"}, b"hello", 400, "letters and digits"),
                ({"voice": "missing"}, b"hello", 400, "unknown voice"),
                ({"speed": "99"}, b"hello", 400, "speed"),
                ({"content-length": "-1"}, b"hello", 400, "content-length"),
                ({}, b"\xff", 400, "valid UTF-8"),
            )
            for headers, body, status_code, detail in cases:
                with self.subTest(headers=headers, body=body):
                    response = client.post(
                        "/v1/audio/speech",
                        content=body,
                        headers=headers,
                    )
                    self.assertEqual(response.status_code, status_code)
                    self.assertIn(detail, response.json()["detail"])

    def test_concurrent_speech_requests_return_success_and_busy(self):
        with self._make_client() as client:

            def send_request():
                return client.post("/v1/audio/speech", content=b"hello")

            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda _: send_request(), range(2)))

        statuses = sorted(response.status_code for response in responses)
        self.assertEqual(statuses, [200, 429])
        rejected = next(response for response in responses if response.status_code == 429)
        self.assertIn("busy", rejected.json()["detail"].lower())

    def test_voice_routes_reject_while_speech_request_is_active(self):
        with self._make_client() as client:
            with ThreadPoolExecutor(max_workers=1) as executor:
                speech = executor.submit(
                    client.post,
                    "/v1/audio/speech",
                    content=("word " * 1_000).encode(),
                )
                time.sleep(0.01)
                voices = client.get("/v1/voices")

            self.assertEqual(speech.result().status_code, 200)
            self.assertEqual(voices.status_code, 429)
            self.assertIn("already handling", voices.json()["detail"])

    def _assert_first_audio_event_is_pcm(self, body: str) -> None:
        marker = "event: audio\ndata: "
        start = body.index(marker) + len(marker)
        end = body.index("\n\n", start)
        pcm = base64.b64decode(body[start:end], validate=True)
        self.assertGreater(len(pcm), 0)
        self.assertEqual(len(pcm) % 2, 0)


class RouterErrorMappingTests(unittest.TestCase):
    def test_as_http_error_maps_known_errors(self):
        passthrough = HTTPException(status_code=418, detail="teapot")
        self.assertIs(_as_http_error(passthrough), passthrough)
        self.assertEqual(_as_http_error(PayloadTooLargeError("too big")).status_code, 413)
        self.assertEqual(_as_http_error(InvalidRequestError("bad")).status_code, 400)
        self.assertEqual(_as_http_error(AdmissionRejectedError("busy")).status_code, 429)
        self.assertEqual(_as_http_error(ProgressTimeoutError("slow")).status_code, 504)
        self.assertEqual(_as_http_error(ComponentLifecycleError("stopped")).status_code, 503)
        self.assertEqual(_as_http_error(RuntimeError("boom")).status_code, 503)
