"""Tests for the public LLM component API."""

from __future__ import annotations

import asyncio
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
from trillim.components.llm._engine import EngineCrashedError
from trillim.components.llm._session import _ChatSession
from trillim.components.llm.public import LLM
from trillim.errors import AdmissionRejectedError, ComponentLifecycleError
from trillim.harnesses.search._harness import _SearchHarness
from tests.components.llm.support import (
    FakeEngine,
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

    async def test_stop_during_start_does_not_restore_running_runtime(self):
        self._ensure_store_dir("Trillim/one")
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: make_runtime_model(
                Path("/tmp/model-one"),
                name="model-one",
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        original_finish_swapping = llm._admission.finish_swapping

        async def blocked_finish_swapping() -> None:
            entered.set()
            await release.wait()
            await original_finish_swapping()

        llm._admission.finish_swapping = blocked_finish_swapping
        start_task = asyncio.create_task(llm.start())
        await entered.wait()
        stop_task = asyncio.create_task(llm.stop())

        try:
            await asyncio.sleep(0.05)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            release.set()
            start_result, stop_result = await asyncio.gather(
                start_task,
                stop_task,
                return_exceptions=True,
            )
            self.assertIsInstance(start_result, ComponentLifecycleError)
            self.assertIsNone(stop_result)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            self.assertIsNone(llm.model_name)
        finally:
            release.set()
            await asyncio.gather(start_task, stop_task, return_exceptions=True)

    async def test_stale_stop_cleanup_does_not_clobber_new_runtime(self):
        self._ensure_store_dir("Trillim/one")
        factory = FakeEngineFactory(responses=["ok"])
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: make_runtime_model(
                Path("/tmp/model-one"),
                name="model-one",
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        await llm.start()
        stale_runtime = llm._build_runtime(
            llm._configured_init_config,
            harness_name=llm._configured_harness_name,
            search_provider=llm._configured_search_provider,
            search_token_budget=llm._configured_search_token_budget,
        )

        try:
            await stale_runtime.engine.start()
            await llm._discard_runtime_after_stop(stale_runtime)
            self.assertEqual(llm.state, LLMState.RUNNING)
            self.assertEqual(llm.model_info().state, LLMState.RUNNING)
            self.assertTrue(llm._admission.accepting)
            self.assertEqual(await llm.chat([{"role": "user", "content": "hi"}]), "ok")
            self.assertEqual(stale_runtime.engine.stop_calls, 1)
        finally:
            if llm.state == LLMState.RUNNING:
                await llm.stop()

    async def test_stop_waits_for_inflight_startup_cleanup(self):
        self._ensure_store_dir("Trillim/one")
        entered = asyncio.Event()
        release = asyncio.Event()

        class _BlockingEngine(FakeEngine):
            async def start(self) -> None:
                self.start_calls += 1
                entered.set()
                await release.wait()

        class _BlockingEngineFactory:
            def __init__(self) -> None:
                self.instances: list[_BlockingEngine] = []

            def __call__(self, model, tokenizer, defaults, **kwargs):
                engine = _BlockingEngine(
                    model,
                    tokenizer,
                    defaults,
                    responses=["ok"],
                    **kwargs,
                )
                self.instances.append(engine)
                return engine

        factory = _BlockingEngineFactory()
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: make_runtime_model(
                Path("/tmp/model-one"),
                name="model-one",
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        start_task = asyncio.create_task(llm.start())
        await entered.wait()
        stop_task = asyncio.create_task(llm.stop())

        try:
            await asyncio.sleep(0.05)
            self.assertFalse(stop_task.done())

            release.set()
            start_result, stop_result = await asyncio.gather(
                start_task,
                stop_task,
                return_exceptions=True,
            )

            self.assertIsInstance(start_result, ComponentLifecycleError)
            self.assertIsNone(stop_result)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            self.assertIsNone(llm.model_name)
            self.assertEqual(factory.instances[0].stop_calls, 1)
        finally:
            release.set()
            await asyncio.gather(start_task, stop_task, return_exceptions=True)

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

    async def test_concurrent_swap_requests_during_preflight_fail_fast_instead_of_queueing(self):
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        self._ensure_store_dir("Trillim/three")
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
            make_runtime_model(Path("/tmp/model-three"), name="model-three"),
        ]
        model_iter = iter(models)
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        await llm.start()
        entered = asyncio.Event()
        release = asyncio.Event()

        class _GateLock:
            def __init__(self) -> None:
                self._lock = asyncio.Lock()
                self._first = True

            async def __aenter__(self):
                await self._lock.acquire()
                if self._first:
                    self._first = False
                    entered.set()
                    await release.wait()
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb
                self._lock.release()

        llm._swap_lock = _GateLock()
        first = asyncio.create_task(llm.swap_model("Trillim/two"))
        await entered.wait()
        second = asyncio.create_task(llm.swap_model("Trillim/three"))

        try:
            await asyncio.sleep(0.05)
            if not second.done():
                self.fail("concurrent swap requests should fail fast instead of queueing")
            self.assertFalse(second.cancelled())
            self.assertIsNotNone(second.exception())
        finally:
            if not second.done():
                second.cancel()
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)
            if llm.state == LLMState.RUNNING:
                await llm.stop()

    async def test_stop_during_preflight_swap_does_not_restore_running_runtime(self):
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        model_iter = iter(models)
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        await llm.start()
        entered = asyncio.Event()
        release = asyncio.Event()

        class _GateLock:
            def __init__(self) -> None:
                self._lock = asyncio.Lock()
                self._first = True

            async def __aenter__(self):
                await self._lock.acquire()
                if self._first:
                    self._first = False
                    entered.set()
                    await release.wait()
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb
                self._lock.release()

        llm._swap_lock = _GateLock()
        swap_task = asyncio.create_task(llm.swap_model("Trillim/two"))
        await entered.wait()
        stop_task = asyncio.create_task(llm.stop())

        try:
            await asyncio.sleep(0.05)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            release.set()
            await asyncio.gather(swap_task, stop_task, return_exceptions=True)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            self.assertIsNone(llm.model_name)
        finally:
            release.set()
            await asyncio.gather(swap_task, stop_task, return_exceptions=True)
            if llm.state == LLMState.RUNNING:
                await llm.stop()

    async def test_engine_failure_while_swap_claimed_and_preflight_fails_marks_component_unhealthy(self):
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        models = [
            make_runtime_model(Path("/tmp/model-one"), name="model-one"),
            make_runtime_model(Path("/tmp/model-two"), name="model-two"),
        ]
        model_iter = iter(models)
        llm = LLM(
            "Trillim/one",
            _model_validator=lambda _: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        await llm.start()
        entered = asyncio.Event()
        release = asyncio.Event()

        class _GateLock:
            def __init__(self) -> None:
                self._lock = asyncio.Lock()
                self._first = True

            async def __aenter__(self):
                await self._lock.acquire()
                if self._first:
                    self._first = False
                    entered.set()
                    await release.wait()
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb
                self._lock.release()

        llm._swap_lock = _GateLock()
        swap_task = asyncio.create_task(llm.swap_model("Trillim/two", search_token_budget=0))
        await entered.wait()
        llm._engine.failure = EngineCrashedError("boom")

        try:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await llm.chat([{"role": "user", "content": "hello"}])

            release.set()
            with self.assertRaisesRegex(ValueError, "search_token_budget must be at least 1"):
                await swap_task

            self.assertEqual(llm.state, LLMState.SERVER_ERROR)
            self.assertIsNone(llm.model_name)
            with self.assertRaisesRegex(RuntimeError, "not started"):
                await llm.chat([{"role": "user", "content": "again"}])
        finally:
            release.set()
            await asyncio.gather(swap_task, return_exceptions=True)
            if llm.state == LLMState.RUNNING:
                await llm.stop()

    async def test_stop_during_swap_handoff_does_not_restore_running_runtime(self):
        self._ensure_store_dir("Trillim/one")
        self._ensure_store_dir("Trillim/two")
        model_one_dir = _model_store.store_path_for_id("Trillim/one")

        def validate_model(path: Path) -> object:
            name = Path(str(path)).name
            return make_runtime_model(Path(f"/tmp/model-{name}"), name=f"model-{name}")

        llm = LLM(
            "Trillim/one",
            _model_validator=validate_model,
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        await llm.start()
        entered = asyncio.Event()
        release = asyncio.Event()
        original_finish_swapping = llm._admission.finish_swapping

        async def blocked_finish_swapping() -> None:
            entered.set()
            await release.wait()
            await original_finish_swapping()

        llm._admission.finish_swapping = blocked_finish_swapping
        swap_task = asyncio.create_task(
            llm.swap_model(
                "Trillim/two",
                harness_name="search",
                search_provider="BRAVE_SEARCH",
                search_token_budget=2048,
            )
        )
        await entered.wait()
        stop_task = asyncio.create_task(llm.stop())

        try:
            await asyncio.sleep(0.05)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            release.set()
            swap_result, stop_result = await asyncio.gather(
                swap_task,
                stop_task,
                return_exceptions=True,
            )
            self.assertIsInstance(swap_result, ComponentLifecycleError)
            self.assertIsNone(stop_result)
            self.assertEqual(llm.state, LLMState.UNAVAILABLE)
            self.assertIsNone(llm.model_name)
            self.assertEqual(llm._configured_init_config.model_dir, model_one_dir)
            self.assertEqual(llm._configured_harness_name, "default")
            self.assertEqual(llm._configured_search_provider, "ddgs")
            self.assertEqual(llm._configured_search_token_budget, 1024)

            await llm.start()
            self.assertEqual(llm.model_name, "model-one")
            self.assertEqual(llm._configured_init_config.model_dir, model_one_dir)
            self.assertEqual(llm._configured_harness_name, "default")
            self.assertEqual(llm._configured_search_provider, "ddgs")
            self.assertEqual(llm._configured_search_token_budget, 1024)
        finally:
            release.set()
            await asyncio.gather(swap_task, stop_task, return_exceptions=True)
            if llm.state == LLMState.RUNNING:
                await llm.stop()

    async def test_public_llm_chats_with_real_adapter_overlay(self):
        root = _model_store.store_path_for_id("Trillim/root")
        adapter = _model_store.store_path_for_id("Local/adapter")
        write_model_bundle(root)
        write_adapter_bundle(adapter, model_root=root)

        llm = LLM(
            "Trillim/root",
            lora_dir="Local/adapter",
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["hello"]),
        )

        await llm.start()
        result = await llm.chat([{"role": "user", "content": "Say hi"}])
        info = llm.model_info()

        self.assertEqual(result, "hello")
        self.assertEqual(info.adapter_path, str(adapter))
        self.assertEqual(llm._engine.init_config.lora_dir, adapter)
        await llm.stop()
