from __future__ import annotations

import asyncio
import base64
import json
import unittest

from trillim.components.stt._engine import (
    MAX_WORKER_OUTPUT_BYTES,
    STTEngine,
    STTEngineCrashedError,
    _encode_transcription_request,
    _error_payload,
    _read_response,
    _RESPONSE_HEADER,
)
from trillim.errors import InvalidRequestError


class STTEngineProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_response_reads_typed_payload(self):
        payload = b"transcript"
        stream = asyncio.StreamReader()
        stream.feed_data(_RESPONSE_HEADER.pack(b"T", len(payload)))
        stream.feed_data(payload)

        self.assertEqual(await _read_response(stream), (b"T", payload))

    async def test_read_response_rejects_oversized_payload(self):
        stream = asyncio.StreamReader()
        stream.feed_data(_RESPONSE_HEADER.pack(b"T", MAX_WORKER_OUTPUT_BYTES + 1))

        with self.assertRaisesRegex(STTEngineCrashedError, "oversized"):
            await _read_response(stream)

    def test_encode_transcription_request_base64_encodes_pcm(self):
        payload = _encode_transcription_request(
            pcm=b"\x01\x00",
            conditioning_text="context",
            language="en",
        )

        request = json.loads(payload)
        self.assertEqual(request["command"], "transcribe")
        self.assertEqual(base64.b64decode(request["pcm"], validate=True), b"\x01\x00")
        self.assertEqual(request["conditioning_text"], "context")
        self.assertEqual(request["language"], "en")

    def test_error_payload_uses_exception_type_when_message_is_empty(self):
        self.assertEqual(_error_payload(Exception()), b"Exception")


class STTEngineValidationTests(unittest.TestCase):
    def test_validate_pcm_accepts_bytes_like_values(self):
        engine = STTEngine()
        self.assertEqual(engine._validate_pcm(bytearray(b"\x00\x00")), b"\x00\x00")
        self.assertEqual(engine._validate_pcm(memoryview(b"\x00\x00")), b"\x00\x00")

    def test_validate_pcm_rejects_invalid_values(self):
        engine = STTEngine()
        with self.assertRaisesRegex(InvalidRequestError, "must be bytes"):
            engine._validate_pcm("bad")  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            engine._validate_pcm(b"")
        with self.assertRaisesRegex(InvalidRequestError, "whole 16-bit samples"):
            engine._validate_pcm(b"\x00")

    def test_validate_conditioning_text_and_language_reject_invalid_types(self):
        engine = STTEngine()
        with self.assertRaisesRegex(InvalidRequestError, "conditioning_text"):
            engine._validate_conditioning_text(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidRequestError, "language"):
            engine._validate_language(1)  # type: ignore[arg-type]
