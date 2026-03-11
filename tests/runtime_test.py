# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the Runtime lifecycle manager and sync bridges."""

import unittest

from fastapi import APIRouter

from trillim import Runtime
from trillim.server._component import Component


class LLM(Component):
    def __init__(self, calls: list):
        self.calls = calls
        self.started = False

    def router(self) -> APIRouter:
        return APIRouter()

    async def start(self) -> None:
        self.calls.append("llm.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append("llm.stop")
        self.started = False

    async def chat(self, messages: list[dict]) -> str:
        self.calls.append(("llm.chat", tuple(m["content"] for m in messages)))
        return "reply"

    async def stream_chat(self, messages: list[dict]):
        self.calls.append(("llm.stream_chat", tuple(m["content"] for m in messages)))
        yield "event-1"
        yield "event-2"

    def count_tokens(self, messages: list[dict]) -> int:
        self.calls.append(("llm.count_tokens", len(messages)))
        return len(messages)

    @property
    def max_context_tokens(self) -> int:
        return 128

    def fail(self) -> None:
        raise ValueError("boom")


class Whisper(Component):
    def __init__(self, calls: list):
        self.calls = calls
        self.started = False

    def router(self) -> APIRouter:
        return APIRouter()

    async def start(self) -> None:
        self.calls.append("whisper.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append("whisper.stop")
        self.started = False

    async def transcribe(self, audio_bytes: bytes) -> str:
        self.calls.append(("whisper.transcribe", audio_bytes))
        return audio_bytes.decode()


class TTS(Component):
    def __init__(self, calls: list):
        self.calls = calls
        self.started = False

    def router(self) -> APIRouter:
        return APIRouter()

    async def start(self) -> None:
        self.calls.append("tts.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append("tts.stop")
        self.started = False


class BrokenWhisper(Whisper):
    async def start(self) -> None:
        self.calls.append("whisper.start")
        raise RuntimeError("whisper start failed")


class RuntimeTests(unittest.TestCase):
    def test_runtime_starts_in_order_and_stops_in_reverse(self):
        calls: list = []
        runtime = Runtime(LLM(calls), Whisper(calls), TTS(calls))

        runtime.start()
        runtime.stop()

        self.assertEqual(
            calls,
            [
                "llm.start",
                "whisper.start",
                "tts.start",
                "tts.stop",
                "whisper.stop",
                "llm.stop",
            ],
        )

    def test_runtime_context_manager_and_sync_wrappers(self):
        calls: list = []
        messages = [{"role": "user", "content": "hello"}]

        with Runtime(LLM(calls), Whisper(calls), TTS(calls)) as runtime:
            self.assertTrue(runtime.started)
            self.assertEqual(runtime.llm.chat(messages), "reply")
            self.assertEqual(list(runtime.llm.stream_chat(messages)), ["event-1", "event-2"])
            self.assertEqual(runtime.llm.count_tokens(messages), 1)
            self.assertEqual(runtime.llm.max_context_tokens, 128)
            self.assertEqual(runtime.whisper.transcribe(b"audio"), "audio")

        self.assertFalse(runtime.started)
        self.assertIn(("llm.chat", ("hello",)), calls)
        self.assertIn(("llm.stream_chat", ("hello",)), calls)
        self.assertIn(("llm.count_tokens", 1), calls)
        self.assertIn(("whisper.transcribe", b"audio"), calls)

    def test_runtime_rolls_back_started_components_when_start_fails(self):
        calls: list = []
        runtime = Runtime(LLM(calls), BrokenWhisper(calls), TTS(calls))

        with self.assertRaisesRegex(RuntimeError, "whisper start failed"):
            runtime.start()

        self.assertFalse(runtime.started)
        self.assertEqual(calls, ["llm.start", "whisper.start", "llm.stop"])

    def test_runtime_rejects_duplicate_component_types(self):
        with self.assertRaisesRegex(ValueError, "Duplicate component type"):
            Runtime(LLM([]), LLM([]))

    def test_runtime_method_errors_propagate(self):
        runtime = Runtime(LLM([]))
        runtime.start()
        try:
            with self.assertRaisesRegex(ValueError, "boom"):
                runtime.llm.fail()
        finally:
            runtime.stop()

    def test_runtime_requires_start_before_component_use(self):
        runtime = Runtime(LLM([]))

        with self.assertRaisesRegex(RuntimeError, "Runtime not started"):
            runtime.llm.max_context_tokens


if __name__ == "__main__":
    unittest.main()
