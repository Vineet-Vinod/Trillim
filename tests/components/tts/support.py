from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from trillim.components.tts.public import TTS


class _FakeTokens:
    def __init__(self, count: int) -> None:
        self.shape = (count,)


class _FakeTokenized:
    def __init__(self, count: int) -> None:
        self.tokens = _FakeTokens(count)


class FakeTokenizer:
    def __call__(self, text: str) -> _FakeTokenized:
        return _FakeTokenized(max(1, len(text.split())))


class FakeTTSEngine:
    instances: list[FakeTTSEngine] = []

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.synthesize_calls: list[tuple[str, object]] = []
        self.voice_build_calls: list[Path] = []
        self.synthesize_error: Exception | None = None
        self.build_error: Exception | None = None
        self.synthesize_delay = 0.0
        FakeTTSEngine.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def synthesize_segment(self, text: str, *, voice_state) -> bytes:
        if self.synthesize_error is not None:
            raise self.synthesize_error
        if self.synthesize_delay:
            import asyncio

            await asyncio.sleep(self.synthesize_delay)
        self.synthesize_calls.append((text, voice_state))
        return text.encode("utf-8")

    async def build_voice_state(self, audio_path: str | Path) -> dict:
        if self.build_error is not None:
            raise self.build_error
        self.voice_build_calls.append(Path(audio_path))
        return fake_voice_state()


def fake_voice_state(value: float = 1.0) -> dict:
    return {"module": {"cache": torch.tensor([value])}}


def patched_tts_environment(root: Path):
    stack = ExitStack()
    FakeTTSEngine.instances.clear()
    stack.enter_context(patch("trillim.components.tts.public.TTSEngine", FakeTTSEngine))
    stack.enter_context(
        patch(
            "trillim.components.tts.public._load_built_in_voice_names",
            return_value=("alba", "marius"),
        )
    )
    stack.enter_context(
        patch(
            "trillim.components.tts.public.load_pocket_tts_tokenizer",
            return_value=FakeTokenizer(),
        )
    )
    stack.enter_context(patch("trillim.components.tts.public.VOICE_STORE_ROOT", root))
    stack.enter_context(
        patch(
            "trillim.components.tts.public.importlib.import_module",
            side_effect=lambda name: object(),
        )
    )
    return stack


async def make_started_tts(root: Path) -> tuple[TTS, FakeTTSEngine, ExitStack]:
    stack = patched_tts_environment(root)
    tts = TTS()
    await tts.start()
    return tts, FakeTTSEngine.instances[-1], stack
