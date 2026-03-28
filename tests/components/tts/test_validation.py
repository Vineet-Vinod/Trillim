"""Tests for TTS validation helpers."""

from __future__ import annotations

import errno
import io
import os
import stat
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from trillim.components.tts._limits import MAX_VOICE_STATE_BYTES
from trillim.components.tts._validation import (
    PayloadTooLargeError,
    _validate_content_length,
    _validate_loaded_voice_state_value,
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
        with self.assertRaisesRegex(InvalidRequestError, "must be a string"):
            validate_text(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_text("   ")
        with self.assertRaisesRegex(InvalidRequestError, "character limit"):
            validate_text("x" * 1_000_001)
        self.assertEqual(validate_text("hello"), "hello")

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
        with self.assertRaises(PayloadTooLargeError):
            validate_http_speech_body(b"x" * ((6 * 1024 * 1024) + 1))

    def test_validate_voice_bytes_and_state_bytes_enforce_bounds(self):
        self.assertEqual(validate_voice_bytes(b"abc"), b"abc")
        with self.assertRaisesRegex(InvalidRequestError, "audio must be bytes"):
            validate_voice_bytes("abc")  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_voice_bytes(b"")
        with self.assertRaises(PayloadTooLargeError):
            validate_voice_bytes(b"x" * ((25 * 1024 * 1024) + 1))
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_voice_state_bytes(b"")
        with self.assertRaisesRegex(InvalidRequestError, "byte limit"):
            validate_voice_state_bytes(b"x" * (MAX_VOICE_STATE_BYTES + 1))

    def test_load_safe_voice_state_bytes_rejects_unsafe_or_malformed_payloads(self):
        buffer = io.BytesIO()
        torch.save({"layer": {"cache": torch.tensor([1.0])}}, buffer)
        loaded = load_safe_voice_state_bytes(buffer.getvalue())
        self.assertIn("layer", loaded)

        bad_buffer = io.BytesIO()
        torch.save({"bad": _UnsafeState()}, bad_buffer)
        with self.assertRaisesRegex(InvalidRequestError, "voice state is malformed"):
            load_safe_voice_state_bytes(bad_buffer.getvalue())

        for payload in ({}, {"bad": {1: torch.tensor([1.0])}}, {"bad": {1, 2, 3}}):
            with self.subTest(payload=payload):
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                with self.assertRaisesRegex(InvalidRequestError, "voice state is malformed"):
                    load_safe_voice_state_bytes(buffer.getvalue())

        _validate_loaded_voice_state_value(
            {
                "tensor": torch.tensor([1.0]),
                "nested": [1, "ok", None, (True, 3.5)],
            },
            torch,
        )
        with self.assertRaisesRegex(InvalidRequestError, "voice state is malformed"):
            _validate_loaded_voice_state_value({1: torch.tensor([1.0])}, torch)
        with self.assertRaisesRegex(InvalidRequestError, "voice state is malformed"):
            _validate_loaded_voice_state_value(object(), torch)

    def test_validate_speed_and_name_helpers(self):
        self.assertEqual(validate_speed("2.0"), 2.0)
        self.assertEqual(validate_speed(0.25), 0.25)
        self.assertEqual(validate_speed(4.0), 4.0)
        with self.assertRaisesRegex(InvalidRequestError, "must be a number"):
            validate_speed("fast")
        with self.assertRaisesRegex(InvalidRequestError, "between 0.25 and 4.0"):
            validate_speed("10")
        self.assertEqual(normalize_optional_name(" x ", field_name="voice"), "x")
        self.assertIsNone(normalize_optional_name(None, field_name="voice"))
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            normalize_optional_name("  ", field_name="voice")
        for value in ("bad/name", "../escape", "bad-name", "bad_name", "bad.name"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    InvalidRequestError,
                    "must contain only letters and digits",
                ):
                    normalize_optional_name(value, field_name="voice")
        with self.assertRaisesRegex(InvalidRequestError, "header is required"):
            normalize_required_name(None, field_name="name")
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            normalize_required_name("  ", field_name="name")
        with patch(
            "trillim.components.tts._validation.normalize_optional_name",
            return_value=None,
        ):
            with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
                normalize_required_name("voice", field_name="name")

    def test_validate_http_requests_reject_non_alphanumeric_voice_names(self):
        for value in ("bad/name", "bad-name"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    InvalidRequestError,
                    "must contain only letters and digits",
                ):
                    validate_http_speech_request(
                        content_length=None,
                        voice=value,
                        speed=None,
                    )
                with self.assertRaisesRegex(
                    InvalidRequestError,
                    "must contain only letters and digits",
                ):
                    validate_http_voice_upload_request(
                        content_length=None,
                        name=value,
                    )

    def test_validate_source_audio_path_and_open_reject_symlink_and_non_regular_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "voice.wav"
            source.write_bytes(b"abc")
            self.assertEqual(validate_source_audio_path(source), source)
            fd = open_validated_source_audio_file(source)
            self.assertGreaterEqual(fd, 0)
            os.close(fd)
            symlink = root / "link.wav"
            symlink.symlink_to(source)
            with self.assertRaisesRegex(InvalidRequestError, "must not use symlinks"):
                open_validated_source_audio_file(symlink)
            with self.assertRaisesRegex(InvalidRequestError, "does not exist"):
                open_validated_source_audio_file(root / "missing.wav")

            directory = root / "directory"
            directory.mkdir()
            with self.assertRaisesRegex(InvalidRequestError, "not a regular file"):
                open_validated_source_audio_file(directory)

    def test_validate_source_audio_path_and_internal_helpers_cover_error_matrix(self):
        with self.assertRaisesRegex(InvalidRequestError, "path is required"):
            validate_source_audio_path("")  # type: ignore[arg-type]

        self.assertIsNone(
            _validate_content_length(None, limit=5, kind="payload"),
        )
        self.assertEqual(
            _validate_content_length("5", limit=5, kind="payload"),
            5,
        )
        with self.assertRaisesRegex(InvalidRequestError, "invalid content-length"):
            _validate_content_length("-1", limit=5, kind="payload")
        with self.assertRaisesRegex(InvalidRequestError, "invalid content-length"):
            _validate_content_length("nope", limit=5, kind="payload")
        with self.assertRaisesRegex(PayloadTooLargeError, "payload exceeds"):
            _validate_content_length("6", limit=5, kind="payload")

        fake_path = Path("/tmp/fake.wav")
        fake_stat = types.SimpleNamespace(st_mode=stat.S_IFREG, st_size=4)
        with patch(
            "trillim.components.tts._validation.os.stat",
            side_effect=OSError("boom"),
        ):
            with self.assertRaisesRegex(InvalidRequestError, "could not be opened"):
                open_validated_source_audio_file(fake_path)

        with patch(
            "trillim.components.tts._validation.os.stat",
            return_value=fake_stat,
        ), patch(
            "trillim.components.tts._validation.os.open",
            side_effect=FileNotFoundError("gone"),
        ):
            with self.assertRaisesRegex(InvalidRequestError, "does not exist"):
                open_validated_source_audio_file(fake_path)

        with patch(
            "trillim.components.tts._validation.os.stat",
            return_value=fake_stat,
        ), patch(
            "trillim.components.tts._validation.os.open",
            side_effect=OSError(errno.ELOOP, "loop"),
        ):
            with self.assertRaisesRegex(InvalidRequestError, "must not use symlinks"):
                open_validated_source_audio_file(fake_path)

        with patch(
            "trillim.components.tts._validation.os.stat",
            return_value=fake_stat,
        ), patch(
            "trillim.components.tts._validation.os.open",
            side_effect=OSError(errno.EACCES, "blocked"),
        ):
            with self.assertRaisesRegex(InvalidRequestError, "could not be opened"):
                open_validated_source_audio_file(fake_path)

        with patch(
            "trillim.components.tts._validation.os.stat",
            return_value=fake_stat,
        ), patch(
            "trillim.components.tts._validation.os.open",
            return_value=7,
        ), patch(
            "trillim.components.tts._validation.os.close",
        ) as close:
            with patch(
                "trillim.components.tts._validation.os.fstat",
                return_value=types.SimpleNamespace(st_mode=stat.S_IFDIR, st_size=4),
            ):
                with self.assertRaisesRegex(InvalidRequestError, "not a regular file"):
                    open_validated_source_audio_file(fake_path)
            close.assert_called_once_with(7)

        for size, expected in (
            ((25 * 1024 * 1024) + 1, PayloadTooLargeError),
            (0, InvalidRequestError),
        ):
            with self.subTest(size=size):
                with patch(
                    "trillim.components.tts._validation.os.stat",
                    return_value=fake_stat,
                ), patch(
                    "trillim.components.tts._validation.os.open",
                    return_value=8,
                ), patch(
                    "trillim.components.tts._validation.os.close",
                ) as close:
                    with patch(
                        "trillim.components.tts._validation.os.fstat",
                        return_value=types.SimpleNamespace(st_mode=stat.S_IFREG, st_size=size),
                    ):
                        with self.assertRaises(expected):
                            open_validated_source_audio_file(fake_path)
                    close.assert_called_once_with(8)
