"""Tests for the STT worker subprocess path."""

from __future__ import annotations

import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.stt import _worker
from trillim.components.stt._worker import WorkerFailureError, main, transcribe_owned_audio_file
from trillim.errors import ProgressTimeoutError
from tests.components.stt.support import prepend_pythonpath, python_command, write_fake_faster_whisper_module


class STTWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.audio_path = self.root / "audio.wav"
        self.audio_path.write_bytes(b"hello-audio")

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_transcribe_owned_audio_file_returns_final_text_and_passes_language(self):
        module_dir = self.root / "fake_module"
        module_dir.mkdir()
        write_fake_faster_whisper_module(module_dir)
        record_path = self.root / "record.json"
        env = {
            **prepend_pythonpath(module_dir),
            "TRILLIM_STT_WORKER_RECORD": str(record_path),
            "TRILLIM_STT_WORKER_TEXT": "stub transcript",
        }

        with patch.dict(os.environ, env, clear=False):
            text = await transcribe_owned_audio_file(self.audio_path, language="en")

        self.assertEqual(text, "stub transcript")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["audio_path"], str(self.audio_path))
        self.assertEqual(record["language"], "en")
        self.assertEqual(record["model_name"], "base")
        self.assertEqual(record["device"], "cpu")
        self.assertEqual(record["compute_type"], "int8")
        self.assertEqual(record["audio_bytes"], "hello-audio")

    async def test_transcribe_owned_audio_file_fails_closed_on_malformed_output(self):
        with patch.object(
            _worker,
            "_worker_command",
            return_value=python_command("print('oops')"),
        ):
            with self.assertRaisesRegex(WorkerFailureError, "malformed output"):
                await transcribe_owned_audio_file(self.audio_path, language=None)

    async def test_transcribe_owned_audio_file_fails_closed_on_non_zero_exit(self):
        with patch.object(
            _worker,
            "_worker_command",
            return_value=python_command("raise SystemExit(1)"),
        ):
            with self.assertRaisesRegex(WorkerFailureError, "worker failed"):
                await transcribe_owned_audio_file(self.audio_path, language=None)

    async def test_transcribe_owned_audio_file_times_out_and_kills_worker(self):
        with patch.object(
            _worker,
            "_worker_command",
            return_value=python_command(
                "import signal",
                "import time",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "time.sleep(60)",
            ),
        ), patch.object(_worker, "TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS", 0.01), patch.object(
            _worker,
            "WORKER_KILL_AFTER_SECONDS",
            0.01,
        ):
            with self.assertRaisesRegex(ProgressTimeoutError, "timed out"):
                await transcribe_owned_audio_file(self.audio_path, language=None)

    def test_main_returns_one_on_local_transcription_failure(self):
        with patch(
            "trillim.components.stt._worker._transcribe_locally",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(main(["--audio-path", str(self.audio_path)]), 1)

    def test_main_returns_zero_and_writes_json_on_success(self):
        stdout = io.StringIO()
        with patch(
            "trillim.components.stt._worker._transcribe_locally",
            return_value="hello",
        ), patch("sys.stdout", stdout):
            self.assertEqual(main(["--audio-path", str(self.audio_path)]), 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"text": "hello"})

    def test_parse_worker_output_rejects_missing_text(self):
        with self.assertRaisesRegex(WorkerFailureError, "malformed output"):
            _worker._parse_worker_output(b"{}")

    def test_transcribe_locally_uses_fixed_worker_config(self):
        captured: dict[str, object] = {}

        class _Segment:
            def __init__(self, text: str) -> None:
                self.text = text

        class _WhisperModel:
            def __init__(self, model_name, *, device, compute_type) -> None:
                captured["init"] = (model_name, device, compute_type)

            def transcribe(self, audio_path, language=None):
                captured["call"] = (audio_path, language)
                return ([_Segment("hello"), _Segment(" world")], None)

        fake_module = types.SimpleNamespace(WhisperModel=_WhisperModel)
        with patch.dict("sys.modules", {"faster_whisper": fake_module}):
            text = _worker._transcribe_locally(self.audio_path, language="en")

        self.assertEqual(text, "hello world")
        self.assertEqual(captured["init"], ("base", "cpu", "int8"))
        self.assertEqual(captured["call"], (str(self.audio_path), "en"))

    def test_worker_command_builds_expected_module_invocation(self):
        command = _worker._worker_command(self.audio_path, language="en")
        self.assertEqual(command[1:4], ("-m", "trillim.components.stt._worker", "--audio-path"))
        self.assertEqual(command[-2:], ("--language", "en"))
