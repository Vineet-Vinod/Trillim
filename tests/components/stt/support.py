"""Test helpers for the STT component."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path


class FakeRequest:
    """Minimal request stub for direct STT route handler tests."""

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        chunks: list[bytes | BaseException] | None = None,
    ) -> None:
        self.headers = headers or {}
        self.query_params = query_params or {}
        self._chunks = list(chunks or [])
        self.stream_called = False

    async def stream(self):
        self.stream_called = True
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def make_faster_whisper_stub():
    """Build an in-process faster_whisper stub for STT.start tests."""

    class WhisperModel:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    return types.SimpleNamespace(WhisperModel=WhisperModel)


def list_spool_files(spool_dir: Path) -> list[Path]:
    """Return the STT-owned temp files in one spool directory."""
    if not spool_dir.exists():
        return []
    return sorted(spool_dir.glob("stt-*.audio"))


def prepend_pythonpath(path: Path) -> dict[str, str]:
    """Return a child environment with one extra PYTHONPATH entry."""
    existing = os.environ.get("PYTHONPATH")
    value = str(path) if not existing else f"{path}{os.pathsep}{existing}"
    return {"PYTHONPATH": value}


def write_fake_faster_whisper_module(root: Path) -> None:
    """Write a subprocess-visible faster_whisper stub to one temp directory."""
    module_path = root / "faster_whisper.py"
    module_path.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "from pathlib import Path",
                "",
                "class _Segment:",
                "    def __init__(self, text):",
                "        self.text = text",
                "",
                "class WhisperModel:",
                "    def __init__(self, model_name, *, device='cpu', compute_type='int8'):",
                "        self.model_name = model_name",
                "        self.device = device",
                "        self.compute_type = compute_type",
                "",
                "    def transcribe(self, audio_path, language=None):",
                "        record_path = os.environ.get('TRILLIM_STT_WORKER_RECORD')",
                "        if record_path:",
                "            payload = {",
                "                'audio_path': audio_path,",
                "                'language': language,",
                "                'model_name': self.model_name,",
                "                'device': self.device,",
                "                'compute_type': self.compute_type,",
                "                'audio_bytes': Path(audio_path).read_bytes().decode('latin-1'),",
                "            }",
                "            Path(record_path).write_text(json.dumps(payload), encoding='utf-8')",
                "        mode = os.environ.get('TRILLIM_STT_WORKER_MODE', 'ok')",
                "        if mode == 'raise':",
                "            raise RuntimeError('worker boom')",
                "        text = os.environ.get('TRILLIM_STT_WORKER_TEXT', 'stub transcript')",
                "        return ([_Segment(text)], None)",
            ]
        ),
        encoding="utf-8",
    )


def python_command(*snippets: str) -> tuple[str, ...]:
    """Build a simple child Python command for worker tests."""
    return (sys.executable, "-c", "\n".join(snippets))
