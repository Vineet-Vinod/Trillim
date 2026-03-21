"""Tests for STT validation helpers."""

from __future__ import annotations

import tempfile
import os
import stat
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trillim.components.stt._config import OwnedAudioInput, SourceFileSnapshot
from trillim.components.stt._validation import (
    PayloadTooLargeError,
    open_validated_source_file,
    validate_audio_bytes,
    validate_http_request,
    validate_language,
    validate_owned_audio_input,
    validate_source_file,
    validate_source_snapshot,
)
from trillim.errors import InvalidRequestError


class STTValidationTests(unittest.TestCase):
    def test_validate_language_accepts_hyphenated_codes_and_normalizes_case(self):
        self.assertEqual(validate_language("EN-us"), "en-us")

    def test_validate_language_rejects_invalid_or_overlong_values(self):
        with self.assertRaisesRegex(InvalidRequestError, "letters and hyphens"):
            validate_language("en_us")
        with self.assertRaisesRegex(InvalidRequestError, "character limit"):
            validate_language("a" * 33)

    def test_validate_audio_bytes_rejects_empty_and_oversize_inputs(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_audio_bytes(b"")
        with patch(
            "trillim.components.stt._validation.MAX_UPLOAD_BYTES",
            new=3,
        ):
            with self.assertRaisesRegex(PayloadTooLargeError, "byte limit"):
                validate_audio_bytes(b"abcd")

    def test_validate_source_file_requires_a_path_and_normalizes_it(self):
        with self.assertRaisesRegex(InvalidRequestError, "path is required"):
            validate_source_file("")
        self.assertEqual(validate_source_file("~/audio.wav"), Path("~/audio.wav").expanduser())

    def test_open_validated_source_file_rejects_missing_non_file_and_oversize_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing.wav"
            directory = root / "dir"
            directory.mkdir()
            audio = root / "audio.wav"
            audio.write_bytes(b"abc")
            with self.assertRaisesRegex(InvalidRequestError, "does not exist"):
                open_validated_source_file(missing)
            with self.assertRaisesRegex(InvalidRequestError, "not a regular file"):
                open_validated_source_file(directory)
            with patch(
                "trillim.components.stt._validation.MAX_UPLOAD_BYTES",
                new=2,
            ):
                with self.assertRaisesRegex(PayloadTooLargeError, "byte limit"):
                    open_validated_source_file(audio)

    def test_open_validated_source_file_returns_a_usable_fd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "audio.wav"
            audio.write_bytes(b"abc")
            fd = open_validated_source_file(audio)
            try:
                self.assertEqual(os.read(fd, 3), b"abc")
            finally:
                os.close(fd)

    def test_open_validated_source_file_uses_binary_flag_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "audio.wav"
            audio.write_bytes(b"abc")
            original_open = os.open
            captured_flags: list[int] = []

            def tracked_open(path, flags, *args, **kwargs):
                captured_flags.append(flags)
                return original_open(path, flags, *args, **kwargs)

            with patch.object(
                __import__("trillim.components.stt._validation", fromlist=["os"]).os,
                "O_BINARY",
                0x8000,
                create=True,
            ), patch(
                "trillim.components.stt._validation.os.open",
                side_effect=tracked_open,
            ):
                fd = open_validated_source_file(audio)
            try:
                self.assertEqual(captured_flags, [os.O_RDONLY | 0x8000])
            finally:
                os.close(fd)

    def test_open_validated_source_file_rejects_non_regular_targets_before_open(self):
        fake_stat = SimpleNamespace(st_mode=stat.S_IFIFO, st_size=0)
        with patch(
            "trillim.components.stt._validation.os.stat",
            return_value=fake_stat,
        ), patch(
            "trillim.components.stt._validation.os.open",
            side_effect=AssertionError("os.open should not be called"),
        ):
            with self.assertRaisesRegex(InvalidRequestError, "not a regular file"):
                open_validated_source_file(Path("/tmp/fifo"))

    def test_validate_owned_audio_input_rejects_empty_and_oversize_owned_copies(self):
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_owned_audio_input(OwnedAudioInput(path=Path("/tmp/audio"), size_bytes=0))
        with patch(
            "trillim.components.stt._validation.MAX_UPLOAD_BYTES",
            new=3,
        ):
            with self.assertRaisesRegex(PayloadTooLargeError, "byte limit"):
                validate_owned_audio_input(
                    OwnedAudioInput(path=Path("/tmp/audio"), size_bytes=4)
                )

    def test_validate_source_snapshot_rejects_changed_metadata(self):
        before = SourceFileSnapshot(size_bytes=1, modified_ns=2)
        after = SourceFileSnapshot(size_bytes=2, modified_ns=2)
        with self.assertRaisesRegex(InvalidRequestError, "changed while it was being copied"):
            validate_source_snapshot(before, after)

    def test_validate_http_request_rejects_invalid_headers(self):
        with self.assertRaisesRegex(InvalidRequestError, "content-type"):
            validate_http_request(
                content_type="text/plain",
                content_length=None,
                language=None,
            )
        with self.assertRaisesRegex(InvalidRequestError, "invalid content-length"):
            validate_http_request(
                content_type="audio/wav",
                content_length="abc",
                language=None,
            )
        with self.assertRaisesRegex(InvalidRequestError, "letters and hyphens"):
            validate_http_request(
                content_type="audio/wav",
                content_length=None,
                language="en_us",
            )

    def test_validate_http_request_accepts_allowed_media_types(self):
        request = validate_http_request(
            content_type="Audio/Wav; charset=binary",
            content_length="12",
            language="EN",
        )
        self.assertEqual(request.content_type, "audio/wav")
        self.assertEqual(request.content_length, 12)
        self.assertEqual(request.language, "en")

        octet_stream = validate_http_request(
            content_type="application/octet-stream",
            content_length=None,
            language=None,
        )
        self.assertEqual(octet_stream.content_type, "application/octet-stream")
