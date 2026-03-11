# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Trillim Python SDK — composable server components"""

import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

__all__ = [
    "Server",
    "Runtime",
    "LLM",
    "Whisper",
    "TTS",
    "SentenceChunker",
    "ContextOverflowError",
]


def __getattr__(name: str):
    if name == "LLM":
        from .server._llm import LLM

        return LLM
    if name == "Server":
        from .server._server import Server

        return Server
    if name == "Runtime":
        from .runtime import Runtime

        return Runtime
    if name == "TTS":
        from .server._tts import TTS

        return TTS
    if name == "SentenceChunker":
        from .server._tts import SentenceChunker

        return SentenceChunker
    if name == "Whisper":
        from .server._whisper import Whisper

        return Whisper
    if name == "ContextOverflowError":
        from .errors import ContextOverflowError

        return ContextOverflowError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
