from __future__ import annotations

import os
import errno
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.tts._limits import (
    MAX_HTTP_TEXT_BYTES,
    MAX_INPUT_TEXT_CHARS,
    MAX_VOICE_STATE_BYTES,
    MAX_VOICE_UPLOAD_BYTES,
)
from trillim.components.tts._validation import (
    PayloadTooLargeError,
    _flatten_safetensors_voice_state,
    _unflatten_safetensors_voice_state,
    _validate_loaded_voice_state_value,
    dump_voice_state_safetensors_bytes,
    load_safe_voice_state_safetensors,
    load_safe_voice_state_safetensors_bytes,
    normalize_optional_name,
    normalize_required_name,
    open_validated_source_audio_file,
    save_voice_state_safetensors,
    validate_http_speech_body,
    validate_http_speech_request,
    validate_http_voice_upload_request,
    validate_speed,
    validate_source_audio_path,
    validate_text,
    validate_voice_bytes,
    validate_voice_state_bytes,
)
from trillim.errors import InvalidRequestError

from tests.components.tts.support import sample_voice_state


class TTSValidationTests(unittest.TestCase):
    def test_text_name_speed_and_body_validation(self):
        self.assertEqual(validate_text(" hello "), " hello ")
        self.assertEqual(normalize_required_name(" custom ", field_name="voice"), "custom")
        self.assertIsNone(normalize_optional_name(None, field_name="voice"))
        self.assertEqual(validate_speed("1.5"), 1.5)
        self.assertEqual(validate_http_speech_body("hi".encode()), "hi")

        invalid_cases = (
            (validate_text, (123,), "text must be a string"),
            (validate_text, (" ",), "must not be empty"),
            (validate_text, ("x" * (MAX_INPUT_TEXT_CHARS + 1),), "character limit"),
            (normalize_required_name, (None,), "header is required"),
            (normalize_required_name, (" ",), "must not be empty"),
            (normalize_optional_name, ("bad-name",), "letters and digits"),
            (validate_speed, ("fast",), "number"),
            (validate_speed, (99,), "between"),
            (validate_http_speech_body, (b"x" * (MAX_HTTP_TEXT_BYTES + 1),), "byte limit"),
            (validate_http_speech_body, (b"",), "must not be empty"),
            (validate_http_speech_body, (b"\xff",), "valid UTF-8"),
        )
        for func, args, message in invalid_cases:
            with self.subTest(func=func.__name__, message=message):
                kwargs = (
                    {"field_name": "voice"}
                    if func in {normalize_required_name, normalize_optional_name}
                    else {}
                )
                with self.assertRaisesRegex(InvalidRequestError, message):
                    func(*args, **kwargs)

    def test_http_metadata_and_size_validation(self):
        speech = validate_http_speech_request(
            content_length="2",
            voice="alba",
            speed=None,
        )
        self.assertEqual(speech.content_length, 2)
        self.assertEqual(speech.voice, "alba")
        self.assertEqual(speech.speed, 1.0)
        upload = validate_http_voice_upload_request(
            content_length="3",
            name="custom",
        )
        self.assertEqual(upload.content_length, 3)
        self.assertEqual(upload.name, "custom")
        self.assertEqual(validate_voice_bytes(b"voice"), b"voice")

        with self.assertRaisesRegex(InvalidRequestError, "invalid content-length"):
            validate_http_speech_request(content_length="bad", voice=None, speed=None)
        with self.assertRaisesRegex(PayloadTooLargeError, "speech input exceeds"):
            validate_http_speech_request(
                content_length=str(MAX_HTTP_TEXT_BYTES + 1),
                voice=None,
                speed=None,
            )
        with self.assertRaisesRegex(PayloadTooLargeError, "voice upload exceeds"):
            validate_http_voice_upload_request(
                content_length=str(MAX_VOICE_UPLOAD_BYTES + 1),
                name="custom",
            )
        with self.assertRaisesRegex(InvalidRequestError, "audio must be bytes"):
            validate_voice_bytes("voice")
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_voice_bytes(b"")
        with self.assertRaisesRegex(PayloadTooLargeError, "voice upload exceeds"):
            validate_voice_bytes(b"x" * (MAX_VOICE_UPLOAD_BYTES + 1))

    def test_safetensors_voice_state_roundtrip_and_malformed_payload(self):
        payload = dump_voice_state_safetensors_bytes(sample_voice_state())
        state = load_safe_voice_state_safetensors_bytes(payload)
        self.assertEqual(state["module"]["cache"].tolist(), [1.0])

        self.assertEqual(validate_voice_state_bytes(payload), payload)
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_voice_state_bytes(b"")
        with self.assertRaisesRegex(InvalidRequestError, "byte limit"):
            validate_voice_state_bytes(b"x" * (MAX_VOICE_STATE_BYTES + 1))
        with self.assertRaisesRegex(InvalidRequestError, "malformed"):
            load_safe_voice_state_safetensors_bytes(b"not safetensors")
        with self.assertRaisesRegex(InvalidRequestError, "malformed"):
            dump_voice_state_safetensors_bytes({"module": {"bad/key": object()}})

    def test_safetensors_path_validation_and_cleanup_error_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing.safetensors"
            empty = root / "empty.safetensors"
            empty.write_bytes(b"")

            with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                load_safe_voice_state_safetensors(missing)
            with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                load_safe_voice_state_safetensors(empty)
            with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                save_voice_state_safetensors({}, root / "state.safetensors")
            with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                save_voice_state_safetensors("bad", root / "state.safetensors")
            with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                save_voice_state_safetensors(sample_voice_state(), root)

        payload = dump_voice_state_safetensors_bytes(sample_voice_state())
        with patch.object(Path, "unlink", side_effect=OSError("cleanup")):
            state = load_safe_voice_state_safetensors_bytes(payload)
        self.assertEqual(state["module"]["cache"].tolist(), [1.0])

        with patch.object(Path, "unlink", side_effect=OSError("cleanup")):
            dumped = dump_voice_state_safetensors_bytes(sample_voice_state())
        self.assertGreater(len(dumped), 0)

    def test_loaded_voice_state_value_validation(self):
        import torch

        _validate_loaded_voice_state_value(
            {"module": [torch.tensor([1.0]), ("ok", 1, 1.5, True, None)]},
            torch,
        )
        invalid_values = (
            {1: torch.tensor([1.0])},
            {"module": object()},
        )
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                    _validate_loaded_voice_state_value(value, torch)

    def test_safetensors_flatten_and_unflatten_reject_malformed_shapes(self):
        import torch

        flatten_cases = (
            {"bad/module": {"cache": torch.tensor([1.0])}},
            {"module": torch.tensor([1.0])},
            {"module": {"bad/key": torch.tensor([1.0])}},
            {"module": {"cache": object()}},
            {"module": {}},
        )
        for state in flatten_cases:
            with self.subTest(state=repr(state)):
                with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                    _flatten_safetensors_voice_state(state, torch)

        unflatten_cases = (
            {},
            {"bad": torch.tensor([1.0])},
            {"/key": torch.tensor([1.0])},
            {"module/": torch.tensor([1.0])},
            {"module/key": object()},
        )
        for flat_state in unflatten_cases:
            with self.subTest(flat_state=repr(flat_state)):
                with self.assertRaisesRegex(InvalidRequestError, "malformed"):
                    _unflatten_safetensors_voice_state(flat_state, torch)

        self.assertEqual(
            _unflatten_safetensors_voice_state({"module/key": torch.tensor([1.0])}, torch)[
                "module"
            ]["key"].tolist(),
            [1.0],
        )

    def test_open_validated_source_audio_file_accepts_regular_nonempty_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "voice.wav"
            source.write_bytes(b"voice")
            fd = open_validated_source_audio_file(source)
            try:
                self.assertGreater(os.fstat(fd).st_size, 0)
            finally:
                os.close(fd)

    def test_open_validated_source_audio_file_rejects_unsafe_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing.wav"
            empty = root / "empty.wav"
            directory = root / "directory"
            empty.write_bytes(b"")
            directory.mkdir()

            cases = (
                (missing, "does not exist"),
                (empty, "must not be empty"),
                (directory, "not a regular file"),
            )
            for path, message in cases:
                with self.subTest(path=path):
                    with self.assertRaisesRegex(InvalidRequestError, message):
                        open_validated_source_audio_file(path)

            target = root / "target.wav"
            target.write_bytes(b"voice")
            symlink = root / "link.wav"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(InvalidRequestError, "symlinks"):
                open_validated_source_audio_file(symlink)

    def test_source_audio_path_and_open_error_branches(self):
        with self.assertRaisesRegex(InvalidRequestError, "path is required"):
            validate_source_audio_path("")
        self.assertEqual(validate_source_audio_path("~/voice.wav"), Path("~/voice.wav").expanduser())

        class EmptyRawPath:
            _raw_paths = [""]

        with self.assertRaisesRegex(InvalidRequestError, "path is required"):
            validate_source_audio_path(EmptyRawPath())

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "voice.wav"
            source.write_bytes(b"voice")
            with patch("trillim.components.tts._validation.os.stat", side_effect=OSError("stat")):
                with self.assertRaisesRegex(InvalidRequestError, "could not be opened"):
                    open_validated_source_audio_file(source)
            with patch("trillim.components.tts._validation.os.open", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(InvalidRequestError, "does not exist"):
                    open_validated_source_audio_file(source)
            with patch(
                "trillim.components.tts._validation.os.open",
                side_effect=OSError(errno.ELOOP, "loop"),
            ):
                with self.assertRaisesRegex(InvalidRequestError, "symlinks"):
                    open_validated_source_audio_file(source)
            with patch(
                "trillim.components.tts._validation.os.open",
                side_effect=OSError(errno.EACCES, "denied"),
            ):
                with self.assertRaisesRegex(InvalidRequestError, "could not be opened"):
                    open_validated_source_audio_file(source)

            fd = os.open(source, os.O_RDONLY)
            try:
                too_large = os.stat_result(
                    (stat.S_IFREG, 0, 0, 0, 0, 0, MAX_VOICE_UPLOAD_BYTES + 1, 0, 0, 0)
                )
                with patch("trillim.components.tts._validation.os.open", return_value=fd):
                    with patch("trillim.components.tts._validation.os.fstat", return_value=too_large):
                        with self.assertRaisesRegex(PayloadTooLargeError, "voice upload exceeds"):
                            open_validated_source_audio_file(source)
                fd = os.open(source, os.O_RDONLY)
                fake_stat = os.stat_result((stat.S_IFDIR, 0, 0, 0, 0, 0, 1, 0, 0, 0))
                with patch("trillim.components.tts._validation.os.open", return_value=fd):
                    with patch("trillim.components.tts._validation.os.fstat", return_value=fake_stat):
                        with self.assertRaisesRegex(InvalidRequestError, "not a regular file"):
                            open_validated_source_audio_file(source)
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass

        with self.assertRaisesRegex(InvalidRequestError, "invalid content-length"):
            validate_http_speech_request(content_length="-1", voice=None, speed=None)
        self.assertIsNone(
            validate_http_speech_request(content_length=None, voice=None, speed=None).content_length
        )

        with patch(
            "trillim.components.tts._validation.normalize_optional_name",
            return_value=None,
        ):
            with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
                normalize_required_name("custom", field_name="voice")
