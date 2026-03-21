"""Tests for the base harness abstraction."""

from __future__ import annotations

from pathlib import Path
import unittest

from trillim.components.llm._config import SamplingDefaults
from trillim.components.llm._events import ChatFinalTextEvent, ChatTokenEvent
from trillim.harnesses.base import Harness
from tests.components.llm.support import FakeEngine, FakeTokenizer, make_runtime_model


class _ProbeHarness(Harness):
    async def stream_events(self, session, **sampling):
        yield ChatTokenEvent(text="a")
        yield ChatFinalTextEvent(text="ab")


class HarnessBaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_text_yields_only_token_fragments(self):
        engine = FakeEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            defaults=SamplingDefaults(),
        )
        harness = _ProbeHarness(engine)

        chunks = [chunk async for chunk in harness.stream_text(object())]

        self.assertEqual(chunks, ["a"])
        self.assertEqual(harness.completion_tokens, 0)
