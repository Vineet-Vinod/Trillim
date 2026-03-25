"""Test helpers for the TTS component."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from trillim.components.tts.public import TTS


class FakeRequest:
    """Minimal request stub for direct TTS route handler tests."""

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes | BaseException] | None = None,
    ) -> None:
        self.headers = headers or {}
        self._chunks = list(chunks or [])
        self.state = SimpleNamespace()
        self.stream_called = False

    async def stream(self):
        self.stream_called = True
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


class _FakeTokens:
    def __init__(self, count: int) -> None:
        self.shape = (1, count)


class FakeTokenizer:
    """Token counter that approximates PocketTTS token counts by words."""

    def __call__(self, text: str):
        words = [word for word in text.replace("\n", " ").split(" ") if word]
        return SimpleNamespace(tokens=_FakeTokens(len(words)))


async def fake_synthesizer(text: str, *, voice_kind: str, voice_reference: str) -> bytes:
    """Return deterministic PCM bytes for one fake synthesis chunk."""
    return f"{voice_kind}:{voice_reference}:{text}".encode("utf-8")


async def fake_voice_state_builder(audio_path: Path) -> bytes:
    """Return deterministic serialized voice-state bytes."""
    buffer = io.BytesIO()
    torch.save(
        {"prompt": Path(audio_path).read_bytes().decode("latin-1")},
        buffer,
    )
    return buffer.getvalue()


def make_started_tts(
    *,
    default_voice: str = "alba",
    speed: float = 1.0,
    synth=fake_synthesizer,
    voice_state_builder=fake_voice_state_builder,
) -> tuple[TTS, patch, patch]:
    """Create one TTS instance plus dependency patches for start()."""
    class _FakeSessionWorker:
        def __init__(self, *, voice_kind: str, voice_reference: str) -> None:
            self._voice_kind = voice_kind
            self._voice_reference = voice_reference

        async def synthesize(self, text: str) -> bytes:
            return await synth(
                text,
                voice_kind=self._voice_kind,
                voice_reference=self._voice_reference,
            )

        async def close(self) -> None:
            return None

    tts = TTS(
        default_voice=default_voice,
        speed=speed,
        _tokenizer_loader=lambda: FakeTokenizer(),
        _session_worker_factory=lambda *, voice_kind, voice_reference: _FakeSessionWorker(
            voice_kind=voice_kind,
            voice_reference=voice_reference,
        ),
        _voice_state_builder=voice_state_builder,
    )
    return (
        tts,
        patch("trillim.components.tts.public.importlib.import_module", return_value=object()),
        patch(
            "trillim.components.tts.public._load_built_in_voice_names",
            return_value=("alba", "marius"),
        ),
    )


def write_fake_pocket_tts_package(root: Path) -> None:
    """Write a subprocess-visible fake ``pocket_tts`` package."""
    package_dir = root / "pocket_tts"
    utils_dir = package_dir / "utils"
    package_dir.mkdir(parents=True, exist_ok=True)
    utils_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "class _Model:",
                "    def get_state_for_audio_prompt(self, audio_conditioning):",
                "        text = str(audio_conditioning)",
                "        if Path(text).name == 'deny':",
                "            raise ValueError(",
                "                'If you want access to the model with voice cloning, go to https://huggingface.co/kyutai/pocket-tts and accept the terms, then make sure you\\'re logged in locally with `uvx hf auth login`.'",
                "            )",
                "        if text in {'alba', 'marius'}:",
                "            return {'prompt': text}",
                "        if str(audio_conditioning).endswith('.state'):",
                "            return {'state': 'preloaded'}",
                "        return {'prompt': Path(audio_conditioning).read_text(encoding='latin-1')}",
                "",
                "    def generate_audio(self, model_state, text_to_generate, max_tokens=20):",
                "        if 'boom' in text_to_generate:",
                "            raise RuntimeError('synth failure')",
                "        payload = f\"{model_state}|{text_to_generate}|{max_tokens}\"",
                "        return [0.0, 0.5] if 'tiny' in text_to_generate else [0.0, 0.25, -0.25]",
                "",
                "class TTSModel:",
                "    @staticmethod",
                "    def load_model():",
                "        return _Model()",
            ]
        ),
        encoding="utf-8",
    )
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "utils.py").write_text(
        "PREDEFINED_VOICES = {'alba': 'x', 'marius': 'y'}\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: object) -> None:
    """Write one JSON file with UTF-8 encoding."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def prepend_pythonpath(path: Path) -> dict[str, str]:
    """Return an environment patch with one extra ``PYTHONPATH`` entry."""
    existing = os.environ.get("PYTHONPATH")
    value = str(path) if not existing else f"{path}{os.pathsep}{existing}"
    return {"PYTHONPATH": value}
