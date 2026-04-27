from __future__ import annotations

import io
import math
import wave
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from trillim.components.tts.public import TTS


def sample_voice_state(value: float = 1.0) -> dict:
    return {"module": {"cache": torch.tensor([value])}}


def reference_wav_bytes() -> bytes:
    sample_rate = 24_000
    duration_seconds = 0.5
    samples = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(samples):
            sample = int(32767 * 0.1 * math.sin(2 * math.pi * 220 * index / sample_rate))
            frames.extend(sample.to_bytes(2, "little", signed=True))
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


def write_reference_wav(path: Path) -> None:
    path.write_bytes(reference_wav_bytes())


def tts_voice_store_environment(root: Path) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(patch("trillim.components.tts.public.VOICE_STORE_ROOT", root))
    return stack


async def make_started_tts(root: Path) -> tuple[TTS, ExitStack]:
    stack = tts_voice_store_environment(root)
    tts = TTS()
    await tts.start()
    return tts, stack
