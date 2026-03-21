"""Tests for the default harness."""

from __future__ import annotations

from pathlib import Path
import unittest

from trillim.components.llm._config import SamplingDefaults
from trillim.harnesses.default import DefaultHarness
from tests.components.llm.support import FakeEngine, FakeTokenizer, make_runtime_model


class _SessionStub:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.final_text = None

    def _prepare_generation(self):
        return list(self.token_ids)

    def _commit_assistant_turn(self, text: str):
        self.final_text = text


class DefaultHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_harness_streams_tokens_and_final_text(self):
        engine = FakeEngine(
            make_runtime_model(Path("/tmp/model")),
            FakeTokenizer(),
            SamplingDefaults(),
            responses=["ok"],
        )
        harness = DefaultHarness(engine)
        session = _SessionStub([1, 2, 3])

        events = [event async for event in harness.stream_events(session, max_tokens=8)]

        self.assertEqual([event.type for event in events], ["token", "token", "final_text"])
        self.assertEqual(events[-1].text, "ok")
        self.assertEqual(harness.completion_tokens, 2)
        self.assertEqual(session.final_text, "ok")
