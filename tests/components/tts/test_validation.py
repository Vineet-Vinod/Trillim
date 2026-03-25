"""Tests for TTS validation helpers."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import torch

from trillim.components.tts._validation import (
    PayloadTooLargeError,
    load_safe_voice_state_bytes,
    normalize_optional_name,
    normalize_required_name,
    open_validated_source_audio_file,
    validate_http_speech_body,
    validate_http_speech_request,
    validate_http_voice_upload_request,
    validate_source_audio_path,
    validate_speed,
    validate_text,
    validate_voice_bytes,
    validate_voice_state_bytes,
)
from trillim.errors import InvalidRequestError


class _UnsafeState:
    pass


class TTSValidationTests(unittest.TestCase):
    def test_validate_text_rejects_empty_and_oversized_inputs(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_text("   ")
        with self.assertRaisesRegex(InvalidRequestError, "character limit"):
            validate_text("x" * 1_000_001)

    def test_validate_http_speech_request_normalizes_metadata(self):
        request = validate_http_speech_request(
            content_length="3",
            voice=" alba ",
            speed="1.5",
        )
        self.assertEqual(request.content_length, 3)
        self.assertEqual(request.voice, "alba")
        self.assertEqual(request.speed, 1.5)

    def test_validate_http_speech_request_rejects_bad_content_length(self):
        with self.assertRaisesRegex(InvalidRequestError, "content-length"):
            validate_http_speech_request(
                content_length="abc",
                voice=None,
                speed=None,
            )
        with self.assertRaises(PayloadTooLargeError):
            validate_http_speech_request(
                content_length=str((6 * 1024 * 1024) + 1),
                voice=None,
                speed=None,
            )

    def test_validate_http_voice_upload_request_requires_name(self):
        with self.assertRaisesRegex(InvalidRequestError, "name header is required"):
            validate_http_voice_upload_request(content_length=None, name=None)
        request = validate_http_voice_upload_request(content_length="2", name=" voice ")
        self.assertEqual(request.name, "voice")

    def test_validate_http_speech_body_rejects_invalid_utf8_and_empty_input(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_http_speech_body(b"")
        with self.assertRaisesRegex(InvalidRequestError, "valid UTF-8"):
            validate_http_speech_body(b"\xff")

    def test_validate_voice_bytes_and_state_bytes_enforce_bounds(self):
        self.assertEqual(validate_voice_bytes(b"abc"), b"abc")
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_voice_bytes(b"")
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_voice_state_bytes(b"")

    def test_load_safe_voice_state_bytes_rejects_unsafe_or_malformed_payloads(self):
        buffer = io.BytesIO()
        torch.save({"layer": {"cache": torch.tensor([1.0])}}, buffer)
        loaded = load_safe_voice_state_bytes(buffer.getvalue())
        self.assertIn("layer", loaded)

        bad_buffer = io.BytesIO()
        torch.save({"bad": _UnsafeState()}, bad_buffer)
        with self.assertRaisesRegex(InvalidRequestError, "voice state is malformed"):
            load_safe_voice_state_bytes(bad_buffer.getvalue())

    def test_validate_speed_and_name_helpers(self):
        self.assertEqual(validate_speed("2.0"), 2.0)
        with self.assertRaisesRegex(InvalidRequestError, "between 0.25 and 4.0"):
            validate_speed("10")
        self.assertEqual(normalize_optional_name(" x ", field_name="voice"), "x")
        self.assertIsNone(normalize_optional_name(None, field_name="voice"))
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            normalize_required_name("  ", field_name="name")

    def test_validate_source_audio_path_and_open_reject_symlink_and_non_regular_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "voice.wav"
            source.write_bytes(b"abc")
            self.assertEqual(validate_source_audio_path(source), source)
            fd = open_validated_source_audio_file(source)
            self.assertGreaterEqual(fd, 0)
            import os

            os.close(fd)
            symlink = root / "link.wav"
            symlink.symlink_to(source)
            with self.assertRaisesRegex(InvalidRequestError, "must not use symlinks"):
                open_validated_source_audio_file(symlink)
