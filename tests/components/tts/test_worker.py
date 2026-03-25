"""Tests for the TTS worker subprocess helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from trillim.components.tts._validation import load_safe_voice_state_bytes
from trillim.errors import ProgressTimeoutError
from trillim.components.tts._worker import (
    WorkerFailureError,
    _VOICE_CLONE_AUTH_ERROR,
    _audio_tensor_to_pcm_bytes,
    _load_worker_state,
    build_voice_state,
    create_session_worker,
    is_voice_cloning_auth_error,
    synthesize_segment,
)
from tests.components.tts.support import prepend_pythonpath, write_fake_pocket_tts_package


class _UnsafeState:
    pass


def _write_counting_fake_pocket_tts_package(root: Path, counter_path: Path) -> None:
    write_fake_pocket_tts_package(root)
    (root / "pocket_tts" / "__init__.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                f"_COUNTER_PATH = Path({str(counter_path)!r})",
                "",
                "class _Model:",
                "    def get_state_for_audio_prompt(self, audio_conditioning):",
                "        text = str(audio_conditioning)",
                "        if text in {'alba', 'marius'}:",
                "            return {'prompt': text}",
                "        if str(audio_conditioning).endswith('.state'):",
                "            return {'state': 'preloaded'}",
                "        return {'prompt': Path(audio_conditioning).read_text(encoding='latin-1')}",
                "",
                "    def generate_audio(self, model_state, text_to_generate, max_tokens=20):",
                "        del model_state, text_to_generate, max_tokens",
                "        return [0.0, 0.25, -0.25]",
                "",
                "class TTSModel:",
                "    @staticmethod",
                "    def load_model():",
                "        count = 0",
                "        if _COUNTER_PATH.exists():",
                "            count = int(_COUNTER_PATH.read_text(encoding='utf-8') or '0')",
                "        _COUNTER_PATH.write_text(str(count + 1), encoding='utf-8')",
                "        return _Model()",
            ]
        ),
        encoding="utf-8",
    )


class TTSWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        write_fake_pocket_tts_package(self.root)

    async def asyncTearDown(self) -> None:
        self._temp_dir.cleanup()

    async def test_synthesize_segment_returns_pcm_bytes(self):
        with patch.dict(os.environ, prepend_pythonpath(self.root)):
            pcm = await synthesize_segment(
                "tiny prompt",
                voice_kind="predefined",
                voice_reference="alba",
            )
        self.assertEqual(len(pcm), 4)

    async def test_build_voice_state_returns_serialized_bytes(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")
        with patch.dict(os.environ, prepend_pythonpath(self.root)):
            state_bytes = await build_voice_state(audio_path)
        loaded = load_safe_voice_state_bytes(state_bytes)
        self.assertEqual(loaded["prompt"], "voice")

    async def test_persistent_session_worker_reuses_loaded_model_across_segments(self):
        counter_path = self.root / "loads.txt"
        _write_counting_fake_pocket_tts_package(self.root, counter_path)
        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        try:
            with patch.dict(os.environ, prepend_pythonpath(self.root)):
                first = await worker.synthesize("first chunk")
                second = await worker.synthesize("second chunk")
        finally:
            await worker.close()
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(counter_path.read_text(encoding="utf-8"), "1")

    async def test_worker_failures_surface_cleanly(self):
        deny_path = self.root / "deny"
        deny_path.write_text("x", encoding="latin-1")
        with patch.dict(os.environ, prepend_pythonpath(self.root)):
            with self.assertRaises(WorkerFailureError):
                await build_voice_state(deny_path)
            with self.assertRaises(WorkerFailureError):
                await synthesize_segment(
                    "boom",
                    voice_kind="predefined",
                    voice_reference="alba",
                )

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        try:
            with patch.dict(os.environ, prepend_pythonpath(self.root)):
                with self.assertRaises(WorkerFailureError):
                    await worker.synthesize("boom")
        finally:
            await worker.close()

    def test_audio_tensor_to_pcm_bytes_and_auth_error_detection(self):
        self.assertEqual(len(_audio_tensor_to_pcm_bytes([0.0, 0.5])), 4)
        self.assertTrue(is_voice_cloning_auth_error(_VOICE_CLONE_AUTH_ERROR.upper()))

    def test_load_worker_state_rejects_unsafe_serialized_state(self):
        good_path = self.root / "voice.state"
        torch.save({"prompt": "voice"}, good_path)
        self.assertEqual(
            _load_worker_state(
                SimpleNamespace(),
                voice_kind="state_file",
                voice_reference=str(good_path),
            )["prompt"],
            "voice",
        )

        bad_path = self.root / "bad.state"
        torch.save({"bad": _UnsafeState()}, bad_path)
        with self.assertRaisesRegex(Exception, "voice state is malformed"):
            _load_worker_state(
                SimpleNamespace(),
                voice_kind="state_file",
                voice_reference=str(bad_path),
            )

    async def test_worker_helpers_bound_stdout_and_stderr(self):
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 9)",
            ),
        ), patch("trillim.components.tts._worker.MAX_PCM_CHUNK_BYTES", 8):
            with self.assertRaisesRegex(WorkerFailureError, "oversized stdout"):
                await synthesize_segment(
                    "tiny prompt",
                    voice_kind="predefined",
                    voice_reference="alba",
                )

        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('x' * 33); sys.exit(1)",
            ),
        ), patch("trillim.components.tts._worker.MAX_WORKER_ERROR_BYTES", 32):
            with self.assertRaisesRegex(WorkerFailureError, "oversized stderr"):
                await build_voice_state(self.root / "voice.txt")

        worker = create_session_worker(
            voice_kind="predefined",
            voice_reference="alba",
        )
        try:
            with patch(
                "trillim.components.tts._worker._worker_command",
                return_value=(
                    sys.executable,
                    "-c",
                    "import sys, struct, time; "
                    "size = struct.unpack('>I', sys.stdin.buffer.read(4))[0]; "
                    "sys.stdin.buffer.read(size); "
                    "sys.stderr.write('x' * 33); "
                    "sys.stderr.flush(); "
                    "time.sleep(1)",
                ),
            ), patch("trillim.components.tts._worker.MAX_WORKER_ERROR_BYTES", 32):
                with self.assertRaisesRegex(WorkerFailureError, "oversized stderr"):
                    await worker.synthesize("hello")
        finally:
            await worker.close()

    async def test_build_voice_state_times_out(self):
        audio_path = self.root / "voice.txt"
        audio_path.write_text("voice", encoding="latin-1")
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import time; time.sleep(1)",
            ),
        ), patch("trillim.components.tts._worker.VOICE_STATE_BUILD_TIMEOUT_SECONDS", 0.01):
            with self.assertRaisesRegex(ProgressTimeoutError, "voice-state build timed out"):
                await build_voice_state(audio_path)

    async def test_persistent_session_worker_times_out_and_rejects_malformed_stdout(self):
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(
                sys.executable,
                "-c",
                "import time; time.sleep(1)",
            ),
        ), patch("trillim.components.tts._worker.PROGRESS_TIMEOUT_SECONDS", 0.01):
            worker = create_session_worker(
                voice_kind="predefined",
                voice_reference="alba",
            )
            try:
                with self.assertRaisesRegex(ProgressTimeoutError, "chunk timed out"):
                    await worker.synthesize("hello")
            finally:
                await worker.close()

        malformed_script = (
            "import sys, struct; "
            "header = sys.stdin.buffer.read(4); "
            "size = struct.unpack('>I', header)[0]; "
            "sys.stdin.buffer.read(size); "
            "sys.stdout.buffer.write(struct.pack('>cI', b'B', 0)); "
            "sys.stdout.buffer.flush()"
        )
        with patch(
            "trillim.components.tts._worker._worker_command",
            return_value=(sys.executable, "-c", malformed_script),
        ):
            worker = create_session_worker(
                voice_kind="predefined",
                voice_reference="alba",
            )
            try:
                with self.assertRaisesRegex(WorkerFailureError, "malformed stdout"):
                    await worker.synthesize("hello")
            finally:
                await worker.close()
