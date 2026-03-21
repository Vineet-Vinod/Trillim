"""Tests for the public LLM component API."""

from __future__ import annotations

from pathlib import Path
import unittest

from trillim.components.llm._config import LLMState
from trillim.components.llm.public import LLM
from trillim.errors import AdmissionRejectedError
from tests.components.llm.support import FakeEngineFactory, FakeTokenizer, make_runtime_model


class PublicLLMTests(unittest.IsolatedAsyncioTestCase):
    def _make_llm(self, *, responses=None):
        return LLM(
            "models/fake",
            _model_validator=lambda path: make_runtime_model(
                Path(f"/tmp/{Path(str(path)).name}"),
                name=Path(str(path)).name,
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=responses or ["ok"]),
        )

    async def test_public_llm_start_stop_chat_and_stream(self):
        llm = self._make_llm(responses=["hello", "world"])
        await llm.start()

        self.assertEqual(llm.model_info().name, "fake")
        self.assertEqual(await llm.chat([{"role": "user", "content": "hi"}]), "hello")
        events = [
            event async for event in llm.stream_chat([{"role": "user", "content": "again"}])
        ]

        self.assertEqual(events[-1].text, "world")
        await llm.stop()
        self.assertEqual(llm.model_info().state.value, "unavailable")

    async def test_open_session_requires_started_runtime(self):
        llm = self._make_llm()

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            llm.open_session()

    async def test_open_session_rejects_when_llm_is_not_running(self):
        llm = self._make_llm()
        await llm.start()

        llm._state = LLMState.DRAINING
        with self.assertRaisesRegex(AdmissionRejectedError, "draining"):
            llm.open_session([{"role": "user", "content": "hello"}])

        llm._state = LLMState.SWAPPING
        with self.assertRaisesRegex(AdmissionRejectedError, "draining"):
            llm.open_session([{"role": "user", "content": "hello"}])

        llm._state = LLMState.SERVER_ERROR
        with self.assertRaisesRegex(RuntimeError, "not running"):
            llm.open_session([{"role": "user", "content": "hello"}])

        await llm.stop()
