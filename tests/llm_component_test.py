# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the LLM component lifecycle, router, and swap logic."""

import asyncio
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trillim.errors import ContextOverflowError
from trillim.events import (
    ChatDoneEvent,
    ChatFinalTextEvent,
    ChatSearchStartedEvent,
    ChatTokenEvent,
    ChatUsage,
)
from trillim.harnesses._default import DefaultHarness
from trillim.harnesses._search import SearchHarness
from trillim.server import LLM
from trillim.server._llm import _stream_chat, _stream_completion
from trillim.server._models import LoadModelResponse, ServerState


class _FakeTokenizer:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(chr(token_id) for token_id in token_ids)


class _AsyncLockStub:
    def __init__(self, locked: bool = False):
        self._locked = locked
        self.entries = 0

    def locked(self) -> bool:
        return self._locked

    async def __aenter__(self):
        self.entries += 1
        self._locked = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._locked = False
        return False


class _ManagedEngine:
    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        max_context_tokens: int = 64,
        start_error: Exception | None = None,
    ):
        self.responses = list(responses or [])
        self.tokenizer = _FakeTokenizer()
        self.arch_config = SimpleNamespace(
            eos_tokens=[0],
            max_position_embeddings=max_context_tokens,
        )
        self._cached_prompt_str = ""
        self._last_cache_hit = 0
        self.lock = _AsyncLockStub()
        self.start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0
        self.generate_calls: list[dict] = []
        self.init = {}

    async def start(self):
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    async def stop(self):
        self.stop_calls += 1

    async def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.responses:
            response = self.responses.pop(0)
            for ch in response:
                yield ord(ch)


class _EngineFactory:
    def __init__(self):
        self.instances: list[_ManagedEngine] = []
        self.start_errors: dict[str, Exception] = {}

    def __call__(self, model_dir, tokenizer, stop_tokens, default_params, **kwargs):
        engine = _ManagedEngine(
            max_context_tokens=kwargs["arch_config"].max_position_embeddings,
            start_error=self.start_errors.get(model_dir),
        )
        engine.tokenizer = tokenizer
        engine.arch_config = kwargs["arch_config"]
        engine.init = {
            "model_dir": model_dir,
            "tokenizer": tokenizer,
            "stop_tokens": stop_tokens,
            "default_params": default_params,
            **kwargs,
        }
        self.instances.append(engine)
        return engine


class LLMInternalTests(unittest.IsolatedAsyncioTestCase):
    def _make_running_llm(self, response: str = "ok", *, max_context_tokens: int = 64) -> LLM:
        llm = LLM("models/fake")
        llm.engine = _ManagedEngine([response], max_context_tokens=max_context_tokens)
        llm.harness = DefaultHarness(llm.engine)
        llm.state = ServerState.RUNNING
        llm.model_name = "fake-model"
        llm._swap_lock = asyncio.Lock()
        return llm

    async def test_require_started_overflow_and_missing_done_event(self):
        llm = LLM("models/fake")

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            llm.max_context_tokens

        llm = self._make_running_llm(max_context_tokens=1)
        with self.assertRaises(ContextOverflowError):
            llm._validate_token_count(1)

        async def broken_stream():
            yield ChatTokenEvent(text="x")

        llm.stream_chat = lambda *args, **kwargs: broken_stream()
        with self.assertRaisesRegex(RuntimeError, "without a done event"):
            await llm._collect_chat([{"role": "user", "content": "hello"}])

    async def test_create_harness_supports_default_and_search(self):
        llm = LLM("models/fake")
        engine = _ManagedEngine(["ok"])

        default_harness = llm._create_harness(engine, "default", "ddgs")
        search_harness = llm._create_harness(engine, "search", "ddgs")

        self.assertIsInstance(default_harness, DefaultHarness)
        self.assertIsInstance(search_harness, SearchHarness)

    async def test_start_and_stop_manage_engine_and_harness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "sample-model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            arch_config = SimpleNamespace(
                eos_tokens=[1, 2],
                max_position_embeddings=128,
            )
            engine_factory = _EngineFactory()

            llm = LLM(str(model_dir), adapter_dir="adapter", num_threads=4)

            with (
                patch("trillim.utils.load_tokenizer", return_value=_FakeTokenizer()),
                patch("trillim.model_arch.ModelConfig.from_config_json", return_value=arch_config),
                patch("trillim.server._llm.load_default_params", return_value={"temperature": 0.2}),
                patch("trillim.server._llm.InferenceEngine", engine_factory),
            ):
                await llm.start()

            self.assertEqual(llm.model_name, "sample-model")
            self.assertIsNotNone(llm._swap_lock)
            self.assertEqual(llm.state, ServerState.RUNNING)
            self.assertIs(llm.engine, engine_factory.instances[0])
            self.assertIsInstance(llm.harness, DefaultHarness)
            self.assertEqual(engine_factory.instances[0].start_calls, 1)
            self.assertEqual(engine_factory.instances[0].init["stop_tokens"], {1, 2})
            self.assertEqual(engine_factory.instances[0].init["adapter_dir"], "adapter")
            self.assertEqual(engine_factory.instances[0].init["num_threads"], 4)

            await llm.stop()

            self.assertEqual(llm.state, ServerState.NO_MODEL)
            self.assertEqual(engine_factory.instances[0].stop_calls, 1)
            self.assertIsNone(llm.harness)

    async def test_stream_helpers_format_sse_chunks(self):
        llm = self._make_running_llm("unused")

        async def stream_events(self, messages, **sampling):
            yield ChatSearchStartedEvent(query="ignored")
            yield ChatTokenEvent(text="O")
            yield ChatFinalTextEvent(text="OK")
            yield ChatDoneEvent(
                text="OK",
                usage=ChatUsage(
                    prompt_tokens=1,
                    completion_tokens=2,
                    total_tokens=3,
                    cached_tokens=0,
                ),
            )

        llm.stream_chat = MethodType(stream_events, llm)

        with (
            patch("trillim.server._llm.make_id", return_value="chat-1"),
            patch("trillim.server._llm.now", return_value=123),
        ):
            chunks = [chunk async for chunk in _stream_chat(llm, [{"role": "user", "content": "hi"}], {}, "chat-model")]

        self.assertEqual(
            chunks,
            [
                'data: {"id": "chat-1", "object": "chat.completion.chunk", "created": 123, "model": "chat-model", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}\n\n',
                'data: {"id": "chat-1", "object": "chat.completion.chunk", "created": 123, "model": "chat-model", "choices": [{"index": 0, "delta": {"content": "O"}, "finish_reason": null}]}\n\n',
                'data: {"id": "chat-1", "object": "chat.completion.chunk", "created": 123, "model": "chat-model", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )

        llm.engine.responses = ["OK"]
        with (
            patch("trillim.server._llm.make_id", return_value="completion-1"),
            patch("trillim.server._llm.now", return_value=456),
        ):
            completion_chunks = [
                chunk async for chunk in _stream_completion(
                    llm,
                    {"token_ids": [1, 2], "temperature": 0.0},
                    "completion-model",
                )
            ]

        self.assertEqual(
            completion_chunks,
            [
                'data: {"id": "completion-1", "object": "text_completion", "created": 456, "model": "completion-model", "choices": [{"index": 0, "text": "O", "finish_reason": null}]}\n\n',
                'data: {"id": "completion-1", "object": "text_completion", "created": 456, "model": "completion-model", "choices": [{"index": 0, "text": "K", "finish_reason": null}]}\n\n',
                'data: {"id": "completion-1", "object": "text_completion", "created": 456, "model": "completion-model", "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )


class LLMSwapTests(unittest.IsolatedAsyncioTestCase):
    def _make_running_llm(self) -> LLM:
        llm = LLM("models/current")
        llm.model_name = "current"
        llm.engine = _ManagedEngine(["current"])
        llm.harness = DefaultHarness(llm.engine)
        llm.state = ServerState.RUNNING
        llm._swap_lock = asyncio.Lock()
        return llm

    def _write_model_dir(self, root: Path, name: str = "model") -> Path:
        model_dir = root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        return model_dir

    async def test_swap_engine_handles_missing_config_and_invalid_harness(self):
        llm = self._make_running_llm()

        result = await llm._swap_engine("/missing/model")
        self.assertEqual(result.status, "error")
        self.assertIn("config.json not found", result.message)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_model_dir(Path(temp_dir))

            with patch.object(llm, "_create_harness", side_effect=ValueError("bad harness")):
                result = await llm._swap_engine(str(model_dir), harness_name="search")

        self.assertEqual(result.status, "error")
        self.assertIn("Invalid harness config", result.message)

    async def test_swap_engine_validates_adapter_files_and_compatibility(self):
        llm = self._make_running_llm()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = self._write_model_dir(root, "model")
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()

            with patch("trillim.model_store.resolve_model_dir", return_value=str(adapter_dir)):
                result = await llm._swap_engine(str(model_dir), adapter_dir="adapter-ref")
            self.assertIn("trillim_config.json", result.message)

            (adapter_dir / "trillim_config.json").write_text("{}", encoding="utf-8")
            with patch("trillim.model_store.resolve_model_dir", return_value=str(adapter_dir)):
                result = await llm._swap_engine(str(model_dir), adapter_dir="adapter-ref")
            self.assertIn("qmodel.lora", result.message)

            (adapter_dir / "qmodel.lora").write_bytes(b"lora")
            from trillim.model_store import AdapterCompatError

            with (
                patch("trillim.model_store.resolve_model_dir", return_value=str(adapter_dir)),
                patch(
                    "trillim.model_store.validate_adapter_model_compat",
                    side_effect=AdapterCompatError("compat mismatch"),
                ),
            ):
                result = await llm._swap_engine(str(model_dir), adapter_dir="adapter-ref")

        self.assertEqual(result.status, "error")
        self.assertIn("compat mismatch", result.message)

    async def test_swap_engine_handles_config_load_and_start_failures(self):
        llm = self._make_running_llm()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_model_dir(Path(temp_dir), "model")

            with patch("trillim.utils.load_tokenizer", side_effect=ValueError("bad config")):
                result = await llm._swap_engine(str(model_dir))
            self.assertEqual(result.status, "error")
            self.assertIn("Failed to load model config", result.message)

            arch_config = SimpleNamespace(
                eos_tokens=[0],
                max_position_embeddings=32,
            )
            engine_factory = _EngineFactory()
            engine_factory.start_errors[str(model_dir)] = RuntimeError("engine boom")

            with (
                patch("trillim.utils.load_tokenizer", return_value=_FakeTokenizer()),
                patch("trillim.model_arch.ModelConfig.from_config_json", return_value=arch_config),
                patch("trillim.server._llm.load_default_params", return_value={"temperature": 0.2}),
                patch("trillim.server._llm.InferenceEngine", engine_factory),
            ):
                result = await llm._swap_engine(str(model_dir))

        self.assertEqual(result.status, "error")
        self.assertIn("Failed to start engine", result.message)
        self.assertEqual(llm.state, ServerState.NO_MODEL)

    async def test_swap_engine_handles_new_harness_failure_and_success(self):
        llm = self._make_running_llm()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = self._write_model_dir(root, "model")
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "trillim_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "qmodel.lora").write_bytes(b"lora")

            arch_config = SimpleNamespace(
                eos_tokens=[7],
                max_position_embeddings=48,
            )
            engine_factory = _EngineFactory()

            harness_calls = []

            def create_harness(engine, harness_name, search_provider):
                harness_calls.append((engine, harness_name, search_provider))
                if len(harness_calls) == 1:
                    return DefaultHarness(engine)
                raise ValueError("broken harness")

            with (
                patch("trillim.model_store.resolve_model_dir", return_value=str(adapter_dir)),
                patch("trillim.model_store.validate_adapter_model_compat", return_value=None),
                patch("trillim.utils.load_tokenizer", return_value=_FakeTokenizer()),
                patch("trillim.model_arch.ModelConfig.from_config_json", return_value=arch_config),
                patch("trillim.server._llm.load_default_params", return_value={"temperature": 0.5}),
                patch("trillim.server._llm.InferenceEngine", engine_factory),
                patch.object(llm, "_create_harness", side_effect=create_harness),
            ):
                result = await llm._swap_engine(
                    str(model_dir),
                    adapter_dir="adapter-ref",
                    harness_name="default",
                )

            self.assertEqual(result.status, "error")
            self.assertIn("Invalid harness config", result.message)
            self.assertEqual(engine_factory.instances[0].stop_calls, 1)
            self.assertIsNone(llm.engine)
            self.assertEqual(llm.state, ServerState.NO_MODEL)

            llm = self._make_running_llm()
            engine_factory = _EngineFactory()

            with (
                patch("trillim.model_store.resolve_model_dir", return_value=str(adapter_dir)),
                patch("trillim.model_store.validate_adapter_model_compat", return_value=None),
                patch("trillim.utils.load_tokenizer", return_value=_FakeTokenizer()),
                patch("trillim.model_arch.ModelConfig.from_config_json", return_value=arch_config),
                patch("trillim.server._llm.load_default_params", return_value={"temperature": 0.5}),
                patch("trillim.server._llm.InferenceEngine", engine_factory),
            ):
                result = await llm._swap_engine(
                    str(model_dir),
                    adapter_dir="adapter-ref",
                    harness_name="search",
                    search_provider="ddgs",
                    num_threads=8,
                    lora_quant="q4",
                    unembed_quant="q8",
                )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.model, "model")
        self.assertEqual(llm.model_name, "model")
        self.assertEqual(llm.state, ServerState.RUNNING)
        self.assertIs(llm.engine, engine_factory.instances[0])
        self.assertIsInstance(llm.harness, SearchHarness)
        self.assertEqual(llm._adapter_dir, str(adapter_dir))
        self.assertEqual(llm._harness_name, "search")
        self.assertEqual(llm._search_provider, "ddgs")
        self.assertEqual(llm._num_threads, 8)
        self.assertEqual(llm._lora_quant, "q4")
        self.assertEqual(llm._unembed_quant, "q8")
        self.assertEqual(engine_factory.instances[0].init["adapter_dir"], str(adapter_dir))
        self.assertEqual(engine_factory.instances[0].init["num_threads"], 8)


class LLMRouterTests(unittest.TestCase):
    def _make_app(self, llm: LLM) -> FastAPI:
        app = FastAPI()
        app.include_router(llm.router())
        return app

    def _make_running_llm(self, response: str = "ok") -> LLM:
        llm = LLM("models/fake")
        llm.engine = _ManagedEngine([response])
        llm.harness = DefaultHarness(llm.engine)
        llm.state = ServerState.RUNNING
        llm.model_name = "fake-model"
        llm._swap_lock = _AsyncLockStub()
        return llm

    def test_models_route_and_load_model_route_states(self):
        llm = LLM("models/fake")
        llm._swap_lock = _AsyncLockStub()

        with TestClient(self._make_app(llm)) as client:
            response = client.get("/v1/models")
        self.assertEqual(response.json(), {"object": "list", "data": []})

        llm = self._make_running_llm()
        with TestClient(self._make_app(llm)) as client:
            response = client.get("/v1/models")
        self.assertEqual(response.json()["data"][0]["id"], "fake-model")

        allowed_root = Path(tempfile.mkdtemp())
        inside_model = allowed_root / "inside"
        inside_model.mkdir()
        outside_model = Path(tempfile.mkdtemp())

        llm = self._make_running_llm()
        with (
            patch("trillim.model_store.resolve_model_dir", side_effect=RuntimeError("missing model")),
            patch("trillim.model_store.MODELS_DIR", allowed_root),
            TestClient(self._make_app(llm)) as client,
        ):
            response = client.post("/v1/models/load", json={"model_dir": "missing"})
        self.assertEqual(response.status_code, 404)

        llm = self._make_running_llm()
        with (
            patch("trillim.model_store.resolve_model_dir", return_value=str(outside_model)),
            patch("trillim.model_store.MODELS_DIR", allowed_root),
            TestClient(self._make_app(llm)) as client,
        ):
            response = client.post("/v1/models/load", json={"model_dir": "outside"})
        self.assertEqual(response.status_code, 403)

        llm = self._make_running_llm()
        llm._swap_lock = _AsyncLockStub(locked=True)
        with (
            patch("trillim.model_store.resolve_model_dir", return_value=str(inside_model)),
            patch("trillim.model_store.MODELS_DIR", allowed_root),
            TestClient(self._make_app(llm)) as client,
        ):
            response = client.post("/v1/models/load", json={"model_dir": "inside"})
        self.assertEqual(response.status_code, 409)

        llm = self._make_running_llm()
        llm._swap_engine = AsyncMock(
            return_value=LoadModelResponse(
                status="error",
                model=llm.model_name,
                recompiled=False,
                message="swap failed",
            )
        )
        with (
            patch("trillim.model_store.resolve_model_dir", return_value=str(inside_model)),
            patch("trillim.model_store.MODELS_DIR", allowed_root),
            TestClient(self._make_app(llm)) as client,
        ):
            response = client.post("/v1/models/load", json={"model_dir": "inside"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(llm.state, ServerState.RUNNING)

        llm = self._make_running_llm()

        async def swap_success(*args, **kwargs):
            llm.state = ServerState.RUNNING
            return LoadModelResponse(status="success", model="inside", recompiled=False)

        llm._swap_engine = AsyncMock(side_effect=swap_success)
        with (
            patch("trillim.model_store.resolve_model_dir", return_value=str(inside_model)),
            patch("trillim.model_store.MODELS_DIR", allowed_root),
            TestClient(self._make_app(llm)) as client,
        ):
            response = client.post(
                "/v1/models/load",
                json={
                    "model_dir": "inside",
                    "adapter_dir": "adapter",
                    "harness": "search",
                    "search_provider": "ddgs",
                    "threads": 6,
                    "lora_quant": "q4",
                    "unembed_quant": "q8",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "inside")
        llm._swap_engine.assert_awaited_once_with(
            str(inside_model),
            adapter_dir="adapter",
            harness_name="search",
            search_provider="ddgs",
            num_threads=6,
            lora_quant="q4",
            unembed_quant="q8",
        )

    def test_chat_completions_route_covers_errors_success_and_streaming(self):
        llm = self._make_running_llm()

        with TestClient(self._make_app(llm)) as client:
            llm.state = ServerState.SWAPPING
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(response.status_code, 503)

            llm.state = ServerState.NO_MODEL
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(response.status_code, 503)

            llm.state = ServerState.RUNNING
            llm._collect_chat = AsyncMock(
                side_effect=ContextOverflowError(5, 4)
            )
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(response.status_code, 400)

            llm._collect_chat = AsyncMock(
                return_value=(
                    "hello",
                    ChatUsage(
                        prompt_tokens=2,
                        completion_tokens=3,
                        total_tokens=5,
                        cached_tokens=1,
                    ),
                )
            )
            with (
                patch("trillim.server._llm.make_id", return_value="chat-id"),
                patch("trillim.server._llm.now", return_value=321),
            ):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "chat-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "id": "chat-id",
                    "object": "chat.completion",
                    "created": 321,
                    "model": "chat-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "delta": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                        "cached_tokens": 1,
                    },
                },
            )

            def overflow_context(messages):
                raise ContextOverflowError(5, 4)

            llm.validate_context = overflow_context
            response = client.post(
                "/v1/chat/completions",
                json={
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            self.assertEqual(response.status_code, 400)

            def ok_context(messages):
                return 2

            async def event_stream(*args, **kwargs):
                yield ChatTokenEvent(text="O")
                yield ChatFinalTextEvent(text="OK")
                yield ChatDoneEvent(
                    text="OK",
                    usage=ChatUsage(1, 2, 3, 0),
                )

            llm.validate_context = ok_context
            llm.stream_chat = MethodType(lambda self, *args, **kwargs: event_stream(), llm)
            with (
                patch("trillim.server._llm.make_id", return_value="stream-id"),
                patch("trillim.server._llm.now", return_value=654),
            ):
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            self.assertEqual(response.status_code, 200)
            self.assertIn("data: [DONE]", response.text)
            self.assertIn('"content": "O"', response.text)

    def test_completions_route_covers_errors_success_and_streaming(self):
        llm = self._make_running_llm()
        llm.engine.responses = ["OK"]

        with TestClient(self._make_app(llm)) as client:
            llm.state = ServerState.SWAPPING
            response = client.post("/v1/completions", json={"prompt": "hi"})
            self.assertEqual(response.status_code, 503)

            llm.state = ServerState.NO_MODEL
            response = client.post("/v1/completions", json={"prompt": "hi"})
            self.assertEqual(response.status_code, 503)

            llm.state = ServerState.RUNNING
            def overflow_prompt(token_count):
                raise ContextOverflowError(token_count, 1)

            llm._validate_token_count = overflow_prompt
            response = client.post("/v1/completions", json={"prompt": "hi"})
            self.assertEqual(response.status_code, 400)

            llm._validate_token_count = lambda token_count: token_count
            llm.engine.responses = ["OK"]
            llm.engine._last_cache_hit = 4
            with (
                patch("trillim.server._llm.make_id", return_value="completion-id"),
                patch("trillim.server._llm.now", return_value=777),
            ):
                response = client.post(
                    "/v1/completions",
                    json={"model": "completion-model", "prompt": "hi"},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "id": "completion-id",
                    "object": "text_completion",
                    "created": 777,
                    "model": "completion-model",
                    "choices": [
                        {"index": 0, "text": "OK", "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 2,
                        "total_tokens": 4,
                        "cached_tokens": 4,
                    },
                },
            )

            llm.engine.responses = ["OK"]
            with (
                patch("trillim.server._llm.make_id", return_value="completion-stream"),
                patch("trillim.server._llm.now", return_value=888),
            ):
                response = client.post(
                    "/v1/completions",
                    json={"stream": True, "prompt": "hi"},
                )
            self.assertEqual(response.status_code, 200)
            self.assertIn("data: [DONE]", response.text)
            self.assertIn('"text": "O"', response.text)


if __name__ == "__main__":
    unittest.main()
