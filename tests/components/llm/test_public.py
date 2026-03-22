"""Tests for the public LLM component API."""

from __future__ import annotations

from pathlib import Path
import typing
import unittest

import trillim.components.llm as llm_exports
import trillim.components.llm.public as llm_public_exports
from trillim.components.llm import ChatSession
from trillim.components.llm._config import LLMState
from trillim.components.llm._session import _ChatSession
from trillim.components.llm.public import LLM
from trillim.errors import AdmissionRejectedError, ComponentLifecycleError
from trillim.harnesses.search._harness import _SearchHarness
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

    def test_chat_session_is_runtime_public(self):
        self.assertTrue(hasattr(llm_exports, "ChatSession"))
        self.assertTrue(hasattr(llm_public_exports, "ChatSession"))

    def test_chat_session_direct_construction_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "use LLM.open_session"):
            ChatSession()
        with self.assertRaisesRegex(TypeError, "use LLM.open_session"):
            _ChatSession()

    def test_chat_session_public_subclassing_is_rejected(self):
        namespace = {"ChatSession": ChatSession}
        with self.assertRaisesRegex(TypeError, "cannot be subclassed publicly"):
            exec(
                "class UserChatSession(ChatSession, _allow_subclass=True):\n"
                "    @property\n"
                "    def state(self):\n"
                "        return 'open'\n"
                "    @property\n"
                "    def messages(self):\n"
                "        return ()\n"
                "    @property\n"
                "    def cached_token_count(self):\n"
                "        return 0\n"
                "    async def __aenter__(self):\n"
                "        return self\n"
                "    async def __aexit__(self, exc_type, exc, tb):\n"
                "        return None\n"
                "    async def close(self):\n"
                "        return None\n"
                "    def add_user(self, content):\n"
                "        return None\n"
                "    def add_system(self, content):\n"
                "        return None\n"
                "    async def chat(self, **kwargs):\n"
                "        return ''\n"
                "    async def stream_chat(self, **kwargs):\n"
                "        if False:\n"
                "            yield None\n",
                namespace,
            )

    def test_open_session_return_type_hints_resolve_at_runtime(self):
        self.assertIs(typing.get_type_hints(LLM.open_session)["return"], ChatSession)

    async def test_open_session_requires_started_runtime(self):
        llm = self._make_llm()

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            llm.open_session()

    async def test_open_session_returns_owner_created_chat_session(self):
        llm = self._make_llm()
        await llm.start()

        session = llm.open_session([{"role": "user", "content": "hello"}])

        self.assertIsInstance(session, ChatSession)
        self.assertIsInstance(session, _ChatSession)
        await session.close()
        await llm.stop()

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

    async def test_swap_model_requires_running_component(self):
        llm = self._make_llm()

        with self.assertRaisesRegex(ComponentLifecycleError, "requires the component to be running"):
            await llm.swap_model("models/next")

        await llm.start()
        await llm.stop()

        with self.assertRaisesRegex(ComponentLifecycleError, "requires the component to be running"):
            await llm.swap_model("models/next")

    async def test_search_harness_binds_and_clamps_runtime_budget(self):
        llm = LLM(
            "models/fake",
            harness_name="search",
            search_provider="BRAVE_SEARCH",
            search_token_budget=2048,
            _model_validator=lambda _: make_runtime_model(
                Path("/tmp/fake-model"),
                name="fake",
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )

        await llm.start()

        self.assertIsInstance(llm._harness, _SearchHarness)
        self.assertEqual(llm._configured_search_provider, "brave")
        self.assertEqual(llm._runtime_search_token_budget, 1024)
        await llm.stop()
