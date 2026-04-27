from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

from trillim.components.stt import STT
from trillim.components.stt._limits import MAX_UPLOAD_BYTES
from trillim.components.stt._router import _as_http_error
from trillim.components.stt._validation import PayloadTooLargeError, validate_http_request
from trillim.errors import ComponentLifecycleError, InvalidRequestError, ProgressTimeoutError
from trillim.server import Server

EXPECTED_PHRASES = (
    "torpedo",
    "russian grand prix",
    "austrian grand prix",
    "british grand prix",
)


class STTRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = Path(__file__).with_name("test.wav")
        self.fixture_bytes = self.fixture_path.read_bytes()

    def _make_server(self) -> Server:
        return Server(STT())

    def test_audio_transcriptions_accepts_wav_and_octet_stream(self):
        with TestClient(self._make_server().app) as client:
            for content_type in ("audio/wav", "audio/x-wav", "application/octet-stream"):
                with self.subTest(content_type=content_type):
                    response = client.post(
                        "/v1/audio/transcriptions",
                        content=self.fixture_bytes,
                        headers={"content-type": content_type},
                    )
                    self.assertEqual(response.status_code, 200)
                    self._assert_expected_transcript(response.json()["text"])

    def test_audio_transcriptions_rejects_invalid_requests(self):
        cases = (
            (
                {"content-type": "text/plain"},
                {},
                self.fixture_bytes,
                400,
                "content-type",
            ),
            (
                {"content-type": "audio/wav"},
                {"language": "en_us"},
                self.fixture_bytes,
                400,
                "letters and hyphens",
            ),
            (
                {"content-type": "audio/wav", "content-length": "bad"},
                {},
                self.fixture_bytes,
                400,
                "content-length",
            ),
            (
                {"content-type": "audio/wav"},
                {},
                b"",
                400,
                "audio_bytes must not be empty",
            ),
            (
                {"content-type": "audio/wav", "content-length": "999999"},
                {},
                self.fixture_bytes,
                400,
                "content-length",
            ),
        )
        with TestClient(self._make_server().app) as client:
            for headers, params, body, status_code, message in cases:
                with self.subTest(headers=headers, params=params, body_len=len(body)):
                    response = client.post(
                        "/v1/audio/transcriptions",
                        params=params,
                        content=body,
                        headers=headers,
                    )
                    self.assertEqual(response.status_code, status_code)
                    self.assertIn(message, response.json()["detail"])

    def test_audio_transcriptions_rejects_shorter_than_claimed_body(self):
        with TestClient(self._make_server().app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                content=self.fixture_bytes,
                headers={
                    "content-type": "audio/wav",
                    "content-length": str(len(self.fixture_bytes) + 1),
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("content-length", response.json()["detail"])

    def test_concurrent_router_requests_return_success_and_busy(self):
        with TestClient(self._make_server().app) as client:
            def send_request():
                return client.post(
                    "/v1/audio/transcriptions",
                    content=self.fixture_bytes,
                    headers={"content-type": "audio/wav"},
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda _: send_request(), range(2)))

        statuses = sorted(response.status_code for response in responses)
        self.assertEqual(statuses, [200, 429])
        successful = next(response for response in responses if response.status_code == 200)
        rejected = next(response for response in responses if response.status_code == 429)
        self._assert_expected_transcript(successful.json()["text"])
        self.assertIn("already handling", rejected.json()["detail"])

    def _assert_expected_transcript(self, text: str) -> None:
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 100)
        lowered = text.lower()
        for phrase in EXPECTED_PHRASES:
            self.assertIn(phrase, lowered)


class ValidationTests(unittest.TestCase):
    def test_validate_http_request_accepts_none_language(self):
        request = validate_http_request(
            content_type="audio/wav",
            content_length=None,
            language=None,
        )
        self.assertIsNone(request.content_length)
        self.assertIsNone(request.language)

    def test_validate_http_request_normalizes_language_and_content_type(self):
        request = validate_http_request(
            content_type="audio/wav; charset=binary",
            content_length="2",
            language=" EN-US ",
        )

        self.assertEqual(request.content_length, 2)
        self.assertEqual(request.language, "en-us")

    def test_validate_http_request_rejects_missing_or_invalid_fields(self):
        cases = (
            (None, None, None, "content-type"),
            ("audio/wav", None, "", "language must not be empty"),
            ("audio/wav", None, "x" * 33, "character limit"),
            ("audio/wav", "-1", None, "content-length"),
        )
        for content_type, content_length, language, message in cases:
            with self.subTest(
                content_type=content_type,
                content_length=content_length,
                language=language,
            ):
                with self.assertRaisesRegex(InvalidRequestError, message):
                    validate_http_request(
                        content_type=content_type,
                        content_length=content_length,
                        language=language,
                    )

    def test_validate_http_request_rejects_oversized_content_length(self):
        with self.assertRaisesRegex(PayloadTooLargeError, "byte limit"):
            validate_http_request(
                content_type="audio/wav",
                content_length=str(MAX_UPLOAD_BYTES + 1),
                language=None,
            )


class RouterErrorMappingTests(unittest.TestCase):
    def test_as_http_error_maps_known_errors(self):
        passthrough = HTTPException(status_code=418, detail="teapot")
        self.assertIs(_as_http_error(passthrough), passthrough)
        self.assertEqual(_as_http_error(PayloadTooLargeError("too big")).status_code, 413)
        self.assertEqual(_as_http_error(InvalidRequestError("bad")).status_code, 400)
        self.assertEqual(_as_http_error(ProgressTimeoutError("slow")).status_code, 504)
        self.assertEqual(_as_http_error(ComponentLifecycleError("stopped")).status_code, 503)
        self.assertEqual(_as_http_error(RuntimeError("boom")).status_code, 503)
