from __future__ import annotations

import base64
import tempfile
import json
import struct
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from trillim.components.tts._engine import (
    TTSEngine,
    _audio_tensor_to_pcm_bytes,
    _encode_synthesis_request,
    _error_payload,
    is_voice_cloning_auth_error,
)
from trillim.components.tts._validation import load_safe_voice_state_safetensors_bytes
from trillim.errors import InvalidRequestError

from tests.components.tts.support import fake_voice_state


class TTSEngineSubprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_worker_synthesizes_built_in_and_custom_voice_state(self):
        engine = TTSEngine()
        await engine.start()
        try:
            built_in_pcm = await engine.synthesize_segment("hello", voice_state="alba")
            self.assertGreater(len(built_in_pcm), 0)
            self.assertEqual(len(built_in_pcm) % 2, 0)

            with tempfile.TemporaryDirectory() as temp_dir:
                source_path = Path(temp_dir) / "voice.wav"
                _write_reference_wav(source_path)
                voice_state = await engine.build_voice_state(source_path)

            self.assertIsInstance(voice_state, dict)
            self.assertTrue(voice_state)

            custom_pcm = await engine.synthesize_segment(
                "hello",
                voice_state=voice_state,
            )
            self.assertGreater(len(custom_pcm), 0)
            self.assertEqual(len(custom_pcm) % 2, 0)
        finally:
            await engine.stop()

    async def test_real_worker_stop_is_idempotent(self):
        engine = TTSEngine()
        await engine.start()
        await engine.stop()
        await engine.stop()


class TTSEngineHelperTests(unittest.TestCase):
    def test_encode_synthesis_request_for_predefined_voice(self):
        payload = json.loads(
            _encode_synthesis_request(text="hello", voice_state="alba")
        )

        self.assertEqual(payload["command"], "synthesize")
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["voice_state"], {"kind": "predefined", "name": "alba"})

    def test_encode_synthesis_request_for_custom_voice_state_dict(self):
        payload = json.loads(
            _encode_synthesis_request(text="hello", voice_state=fake_voice_state())
        )

        self.assertEqual(payload["voice_state"]["kind"], "serialized")
        encoded = payload["voice_state"]["data"].encode("ascii")
        state = load_safe_voice_state_safetensors_bytes(base64.b64decode(encoded))
        self.assertEqual(state["module"]["cache"].tolist(), [1.0])

    def test_encode_synthesis_request_accepts_bytes_like_state(self):
        state_bytes = base64.b64decode(
            json.loads(
                _encode_synthesis_request(text="hello", voice_state=fake_voice_state())
            )["voice_state"]["data"]
        )

        for voice_state in (state_bytes, bytearray(state_bytes), memoryview(state_bytes)):
            with self.subTest(type=type(voice_state).__name__):
                payload = json.loads(
                    _encode_synthesis_request(text="hello", voice_state=voice_state)
                )
                self.assertEqual(payload["voice_state"]["kind"], "serialized")

    def test_encode_synthesis_request_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(InvalidRequestError, "voice_state"):
            _encode_synthesis_request(text="hello", voice_state=object())

    def test_audio_tensor_to_pcm_bytes_clamps_to_int16(self):
        pcm = _audio_tensor_to_pcm_bytes([-2.0, -1.0, 0.0, 0.5, 2.0])

        self.assertEqual(
            struct.unpack("<5h", pcm),
            (-32767, -32767, 0, 16384, 32767),
        )

    def test_audio_tensor_to_pcm_bytes_accepts_tensors(self):
        pcm = _audio_tensor_to_pcm_bytes(torch.tensor([[0.0, 1.0]]))

        self.assertEqual(struct.unpack("<2h", pcm), (0, 32767))

    def test_error_payload_and_auth_error_detection(self):
        self.assertEqual(_error_payload(RuntimeError()), b"RuntimeError")
        self.assertFalse(is_voice_cloning_auth_error("plain failure"))
        self.assertTrue(
            is_voice_cloning_auth_error(
                "ValueError: We could not download the weights for the model with "
                "voice cloning, but you're trying to use voice cloning. Without voice "
                "cloning, you can use our catalog of voices ['alba', 'marius', "
                "'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']. If "
                "you want access to the model with voice cloning, go to "
                "https://huggingface.co/kyutai/pocket-tts and accept the terms, "
                "then make sure you're logged in locally with `uvx hf auth login`"
            )
        )


def _write_reference_wav(path: Path) -> None:
    sample_rate = 24_000
    duration_seconds = 0.5
    samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, samples, endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(path, audio, sample_rate)
