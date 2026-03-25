"""Tests for the LLM HTTP router."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from trillim import _model_store
from trillim.components.llm import ChatUsage
from trillim.components.llm.public import LLM
from trillim.server import Server
from tests.components.llm.support import (
    FakeEngineFactory,
    FakeTokenizer,
    make_runtime_model,
    patched_model_store,
    write_adapter_bundle,
    write_model_bundle,
)


class LLMRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patched_model_store())

    def _ensure_store_dir(self, store_id: str) -> Path:
        path = _model_store.store_path_for_id(store_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_server(self, *, allow_hot_swap: bool = False, responses=None):
        self._ensure_store_dir("Trillim/fake")
        llm = LLM(
            "Trillim/fake",
            _model_validator=lambda path: make_runtime_model(
                Path(f"/tmp/{Path(str(path)).name}"),
                name=Path(str(path)).name,
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=responses or ["ok"]),
        )
        return Server(llm, allow_hot_swap=allow_hot_swap)

    def test_models_route_reports_truthful_model(self):
        with TestClient(self._make_server().app) as client:
            response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["data"][0]["id"], "fake")

    def test_models_route_reports_adapter_runtime_config(self):
        root = _model_store.store_path_for_id("Trillim/root")
        adapter = _model_store.store_path_for_id("Local/adapter")
        write_model_bundle(root)
        write_adapter_bundle(adapter, model_root=root)
        llm = LLM(
            "Trillim/root",
            num_threads=4,
            lora_dir="Local/adapter",
            lora_quant="q4_0",
            unembed_quant="q8_0",
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        server = Server(llm)

        with TestClient(server.app) as client:
            response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"][0]["adapter_path"], str(adapter))
        self.assertEqual(body["data"][0]["init_config"]["num_threads"], 4)
        self.assertEqual(body["data"][0]["init_config"]["lora_quant"], "q4_0")
        self.assertEqual(body["data"][0]["init_config"]["unembed_quant"], "q8_0")

    def test_chat_completions_returns_final_text(self):
        with TestClient(self._make_server(responses=["hello"]).app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Say hi"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "hello",
        )

    def test_chat_completions_reports_current_model_after_swap(self):
        models = [
            make_runtime_model(Path("/tmp/fake"), name="fake"),
            make_runtime_model(Path("/tmp/next"), name="next"),
        ]
        model_iter = iter(models)
        self._ensure_store_dir("Trillim/fake")
        self._ensure_store_dir("Trillim/next")
        llm = LLM(
            "Trillim/fake",
            _model_validator=lambda _path: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["unused"]),
        )
        server = Server(llm, allow_hot_swap=True)

        async def collect_and_swap(*_args, **_kwargs):
            await llm.swap_model("Trillim/next")
            return "hello", ChatUsage(1, 1, 2, 0)

        llm._collect_chat = collect_and_swap

        with TestClient(server.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Say hi"}]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "next")

    def test_chat_completions_streams_sse_chunks(self):
        with TestClient(self._make_server(responses=["hi"]).app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "stream": True,
                },
            ) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn('"delta": {"role": "assistant"}', body)
        self.assertIn('"content": "h"', body)
        self.assertIn("data: [DONE]", body)

    def test_swap_route_is_only_registered_when_enabled(self):
        with TestClient(self._make_server(allow_hot_swap=False).app) as client:
            response = client.post("/v1/models/swap", json={"model_dir": "Trillim/next"})
        self.assertEqual(response.status_code, 404)

        self._ensure_store_dir("Trillim/next")
        with TestClient(self._make_server(allow_hot_swap=True).app) as client:
            response = client.post("/v1/models/swap", json={"model_dir": "Trillim/next"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "next")

    def test_swap_route_accepts_search_runtime_options(self):
        self._ensure_store_dir("Trillim/fake")
        self._ensure_store_dir("Trillim/next")
        llm = LLM(
            "Trillim/fake",
            _model_validator=lambda path: make_runtime_model(
                Path(f"/tmp/{Path(str(path)).name}"),
                name=Path(str(path)).name,
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["ok"]),
        )
        server = Server(llm, allow_hot_swap=True)

        with TestClient(server.app) as client:
            response = client.post(
                "/v1/models/swap",
                json={
                    "model_dir": "Trillim/next",
                    "harness_name": "search",
                    "search_provider": "BRAVE_SEARCH",
                    "search_token_budget": 2048,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(llm._configured_harness_name, "search")
        self.assertEqual(llm._configured_search_provider, "brave")
        self.assertEqual(llm._configured_search_token_budget, 2048)

    def test_swap_route_accepts_init_runtime_options(self):
        root = _model_store.store_path_for_id("Trillim/root")
        next_root = _model_store.store_path_for_id("Trillim/next")
        adapter = _model_store.store_path_for_id("Local/adapter")
        write_model_bundle(root)
        write_model_bundle(next_root)
        write_adapter_bundle(adapter, model_root=root)
        factory = FakeEngineFactory(responses=["ok"])
        llm = LLM(
            "Trillim/root",
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        server = Server(llm, allow_hot_swap=True)

        with TestClient(server.app) as client:
            response = client.post(
                "/v1/models/swap",
                json={
                    "model_dir": "Trillim/next",
                    "num_threads": 7,
                    "lora_dir": "Local/adapter",
                    "lora_quant": "q4_0",
                    "unembed_quant": "q8_0",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["path"], str(next_root))
        self.assertEqual(response.json()["adapter_path"], str(adapter))
        self.assertEqual(response.json()["init_config"]["num_threads"], 7)
        self.assertEqual(factory.instances[-1].init_config.num_threads, 7)
        self.assertEqual(factory.instances[-1].init_config.lora_dir, adapter)

    def test_swap_route_resets_init_runtime_options_to_defaults_when_omitted(self):
        root = _model_store.store_path_for_id("Trillim/root")
        next_root = _model_store.store_path_for_id("Trillim/next")
        adapter = _model_store.store_path_for_id("Local/adapter")
        write_model_bundle(root)
        write_model_bundle(next_root)
        write_adapter_bundle(adapter, model_root=root)
        factory = FakeEngineFactory(responses=["ok"])
        llm = LLM(
            "Trillim/root",
            num_threads=4,
            lora_dir="Local/adapter",
            lora_quant="q4_0",
            unembed_quant="q8_0",
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=factory,
        )
        server = Server(llm, allow_hot_swap=True)

        with TestClient(server.app) as client:
            response = client.post("/v1/models/swap", json={"model_dir": "Trillim/next"})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["adapter_path"])
        self.assertEqual(response.json()["init_config"]["num_threads"], 0)
        self.assertIsNone(response.json()["init_config"]["lora_quant"])
        self.assertIsNone(response.json()["init_config"]["unembed_quant"])
        self.assertEqual(factory.instances[-1].init_config.num_threads, 0)
        self.assertIsNone(factory.instances[-1].init_config.lora_dir)

    def test_swap_route_rejects_unknown_harness_with_400(self):
        self._ensure_store_dir("Trillim/next")
        with TestClient(self._make_server(allow_hot_swap=True).app) as client:
            response = client.post(
                "/v1/models/swap",
                json={"model_dir": "Trillim/next", "harness_name": "bogus"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown harness", response.json()["detail"])

    def test_swap_route_rejects_raw_model_paths_with_400(self):
        with TestClient(self._make_server(allow_hot_swap=True).app) as client:
            response = client.post("/v1/models/swap", json={"model_dir": "/tmp/ignored"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Trillim/<name> or Local/<name>", response.json()["detail"])

    def test_chat_completions_reports_end_of_turn_usage_with_search_harness(self):
        self._ensure_store_dir("Trillim/fake")
        llm = LLM(
            "Trillim/fake",
            harness_name="search",
            search_provider="ddgs",
            _model_validator=lambda path: make_runtime_model(
                Path(f"/tmp/{Path(str(path)).name}"),
                name=Path(str(path)).name,
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(
                responses=["<search>cats</search>", "answer"]
            ),
        )
        server = Server(llm)

        class _SearchStub:
            async def search(self, query: str) -> str:
                self.query = query
                return "curated cat result"

        with TestClient(server.app) as client:
            llm._harness._search = _SearchStub()
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Find cats"}]},
            )
            cached_token_count = llm._engine.cached_token_count

        self.assertEqual(response.status_code, 200)
        usage = response.json()["usage"]
        self.assertEqual(usage["completion_tokens"], len("answer"))
        self.assertGreater(usage["prompt_tokens"], len("Find cats"))
        self.assertEqual(usage["total_tokens"], cached_token_count)
        self.assertEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )
