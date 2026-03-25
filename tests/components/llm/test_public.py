"""Tests for the public LLM component API."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path, PureWindowsPath
import tempfile
import typing
import unittest

from trillim import _model_store
import trillim.components.llm as llm_exports
import trillim.components.llm.public as llm_public_exports
from trillim.components.llm import ChatSession
from trillim.components.llm._config import LLMState
from trillim.components.llm._session import _ChatSession
from trillim.components.llm.public import LLM
from trillim.errors import AdmissionRejectedError, ComponentLifecycleError
from trillim.harnesses.search._harness import _SearchHarness
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
    write_adapter_bundle,
    write_model_bundle,
)


class PublicLLMTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patched_model_store())

    def _ensure_store_dir(self, store_id: str) -> Path:
        path = _model_store.store_path_for_id(store_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_llm(self, *, responses=None):
        self._ensure_store_dir("Trillim/fake")
        return LLM(
            "Trillim/fake",
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
            await llm.swap_model("Trillim/next")

        await llm.start()
        await llm.stop()

        with self.assertRaisesRegex(ComponentLifecycleError, "requires the component to be running"):
            await llm.swap_model("Trillim/next")

    def test_public_llm_rejects_raw_model_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "model_dir: Model IDs must use the form Trillim/<name> or Local/<name>"):
                LLM(
                    Path(temp_dir),
                    _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
                    _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
                    _engine_factory=FakeEngineFactory(responses=["ok"]),
                )

    def test_public_llm_rejects_raw_lora_paths(self):
        self._ensure_store_dir("Trillim/fake")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "lora_dir: Model IDs must use the form Trillim/<name> or Local/<name>"):
                LLM(
                    "Trillim/fake",
                    lora_dir=Path(temp_dir),
                    _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
                    _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
                    _engine_factory=FakeEngineFactory(responses=["ok"]),
                )

    def test_public_llm_accepts_windows_path_store_ids(self):
        root = self._ensure_store_dir("Trillim/fake")
        adapter = self._ensure_store_dir("Local/adapter")

        llm = LLM(
            PureWindowsPath("Trillim/fake"),
            lora_dir=PureWindowsPath("Local/adapter"),
            _model_validator=lambda _: make_runtime_model(Path("/tmp/fake-model")),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )

        self.assertEqual(llm._configured_init_config.model_dir, root)
        self.assertEqual(llm._configured_init_config.lora_dir, adapter)

    async def test_search_harness_binds_and_clamps_runtime_budget(self):
        self._ensure_store_dir("Trillim/fake")
        llm = LLM(
            "Trillim/fake",
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

    async def test_model_info_reports_adapter_runtime_config(self):
        root = _model_store.store_path_for_id("Trillim/root")
        adapter = _model_store.store_path_for_id("Local/adapter")
        write_model_bundle(root)
        write_adapter_bundle(adapter, model_root=root)
        tokenizer_paths: list[Path] = []
        factory = FakeEngineFactory(responses=["ok"])

        def load_fake_tokenizer(path, **_kwargs):
            tokenizer_paths.append(Path(path))
            return FakeTokenizer()

        llm = LLM(
            "Trillim/root",
            num_threads=6,
            lora_dir="Local/adapter\nignored=1",
            lora_quant="q4_0\nignored=1",
            unembed_quant="q8_0\nignored=1",
            _tokenizer_loader=load_fake_tokenizer,
            _engine_factory=factory,
        )

        await llm.start()

        info = llm.model_info()

        self.assertEqual(info.path, str(root))
        self.assertEqual(info.adapter_path, str(adapter))
        self.assertIsNotNone(info.init_config)
        self.assertEqual(info.init_config.num_threads, 6)
        self.assertEqual(info.init_config.lora_quant, "q4_0")
        self.assertEqual(info.init_config.unembed_quant, "q8_0")
        self.assertNotEqual(tokenizer_paths[0], root)
        self.assertEqual(factory.instances[0].init_config.lora_dir, adapter)
        await llm.stop()

    async def test_swap_model_resets_init_runtime_options_to_defaults_when_omitted(self):
        root = _model_store.store_path_for_id("Trillim/root")
        next_root = _model_store.store_path_for_id("Trillim/next")
        adapter = _model_store.store_path_for_id("Local/adapter")
        write_model_bundle(root)
        write_model_bundle(next_root)
        write_adapter_bundle(adapter, model_root=root)
        factory = FakeEngineFactory(responses=["ok"])
        llm = LLM(
            "Trillim/root",
            num_threads=6,
            lora_dir="Local/adapter",
            lora_quant="q4_0",
            unembed_quant="q8_0",
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )

        await llm.start()
        info = await llm.swap_model("Trillim/next")

        self.assertEqual(info.path, str(next_root))
        self.assertIsNone(info.adapter_path)
        self.assertIsNotNone(info.init_config)
        self.assertEqual(info.init_config.num_threads, 0)
        self.assertIsNone(info.init_config.lora_quant)
        self.assertIsNone(info.init_config.unembed_quant)
        self.assertEqual(factory.instances[-1].init_config.num_threads, 0)
        self.assertIsNone(factory.instances[-1].init_config.lora_dir)
        self.assertIsNone(factory.instances[-1].init_config.lora_quant)
        self.assertIsNone(factory.instances[-1].init_config.unembed_quant)
        await llm.stop()
