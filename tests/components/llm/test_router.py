"""Tests for the LLM HTTP router."""

from __future__ import annotations

from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from trillim.components.llm import ChatUsage
from trillim.server import Server
from tests.components.llm.support import FakeEngineFactory, FakeTokenizer, make_runtime_model
from trillim.components.llm.public import LLM


class LLMRouterTests(unittest.TestCase):
    def _make_server(self, *, allow_hot_swap: bool = False, responses=None):
        llm = LLM(
            "models/fake",
            _model_validator=lambda path: make_runtime_model(
                Path(f"/tmp/{Path(str(path)).name}"),
                name=Path(str(path)).name,
            ),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=responses or ["ok"]),
        )
        return Server(llm, allow_llm_hot_swap=allow_hot_swap)

    def test_models_route_reports_truthful_model(self):
        with TestClient(self._make_server().app) as client:
            response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["data"][0]["id"], "fake")

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
        llm = LLM(
            "models/fake",
            _model_validator=lambda _path: next(model_iter),
            _tokenizer_loader=lambda *_args, **_kwargs: FakeTokenizer(),
            _engine_factory=FakeEngineFactory(responses=["unused"]),
        )
        server = Server(llm)

        async def collect_and_swap(*_args, **_kwargs):
            await llm.swap_model("models/next")
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
            response = client.post("/v1/models/swap", json={"model_dir": "models/next"})
        self.assertEqual(response.status_code, 404)

        with TestClient(self._make_server(allow_hot_swap=True).app) as client:
            response = client.post("/v1/models/swap", json={"model_dir": "models/next"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "next")
