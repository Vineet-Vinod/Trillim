from __future__ import annotations

import base64
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

from trillim.components.tts import TTS
from trillim.components.tts._router import _as_http_error
from trillim.components.tts._validation import PayloadTooLargeError
from trillim.errors import AdmissionRejectedError, InvalidRequestError, ProgressTimeoutError
from trillim.server import Server

from tests.components.tts.support import FakeTTSEngine, patched_tts_environment


class TTSRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.voice_root = Path(self._temp_dir.name) / "voices"
        self._stack = patched_tts_environment(self.voice_root)

    def tearDown(self) -> None:
        self._stack.close()
        self._temp_dir.cleanup()

    def _make_client(self) -> TestClient:
        return TestClient(Server(TTS()).app)

    def test_voice_routes_return_expected_json(self):
        with self._make_client() as client:
            self.assertEqual(client.get("/v1/voices").json(), {"voices": ["alba", "marius"]})

            create = client.post("/v1/voices", content=b"voice", headers={"name": "custom"})
            self.assertEqual(create.status_code, 200)
            self.assertEqual(create.json(), {"name": "custom", "status": "created"})
            self.assertEqual(
                client.get("/v1/voices").json(),
                {"voices": ["alba", "marius", "custom"]},
            )

            delete = client.delete("/v1/voices/custom")
            self.assertEqual(delete.status_code, 200)
            self.assertEqual(delete.json(), {"name": "custom", "status": "deleted"})

    def test_audio_speech_streams_sse_audio_and_done(self):
        with self._make_client() as client:
            response = client.post("/v1/audio/speech", content=b"hello")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        body = response.text
        self.assertIn("event: audio\n", body)
        self.assertIn("event: done\n", body)
        self.assertIn(
            f"data: {base64.b64encode(b'  hello').decode('ascii')}",
            body,
        )

    def test_audio_speech_maps_invalid_requests(self):
        with self._make_client() as client:
            cases = (
                ({}, b"", 400, "speech input must not be empty"),
                ({"voice": "bad-name"}, b"hello", 400, "letters and digits"),
                ({"speed": "99"}, b"hello", 400, "speed"),
                ({"content-length": "-1"}, b"hello", 400, "content-length"),
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

    def test_audio_speech_streams_engine_errors_as_sse_error(self):
        with self._make_client() as client:
            FakeTTSEngine.instances[-1].synthesize_error = RuntimeError("engine boom")
            response = client.post("/v1/audio/speech", content=b"hello")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error\n", response.text)
        self.assertIn("engine boom", response.text)

    def test_concurrent_speech_requests_return_success_and_busy(self):
        with self._make_client() as client:
            FakeTTSEngine.instances[-1].synthesize_delay = 0.05

            def send_request():
                return client.post("/v1/audio/speech", content=b"hello")

            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda _: send_request(), range(2)))

        statuses = sorted(response.status_code for response in responses)
        self.assertEqual(statuses, [200, 429])
        rejected = next(response for response in responses if response.status_code == 429)
        self.assertIn("busy", rejected.json()["detail"].lower())


class RouterErrorMappingTests(unittest.TestCase):
    def test_as_http_error_maps_known_errors(self):
        passthrough = HTTPException(status_code=418, detail="teapot")
        self.assertIs(_as_http_error(passthrough), passthrough)
        self.assertEqual(_as_http_error(PayloadTooLargeError("too big")).status_code, 413)
        self.assertEqual(_as_http_error(InvalidRequestError("bad")).status_code, 400)
        self.assertEqual(_as_http_error(AdmissionRejectedError("busy")).status_code, 429)
        self.assertEqual(_as_http_error(ProgressTimeoutError("slow")).status_code, 504)
        self.assertEqual(_as_http_error(RuntimeError("boom")).status_code, 503)
