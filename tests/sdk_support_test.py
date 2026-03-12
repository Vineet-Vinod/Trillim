# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for lightweight SDK support modules."""

import unittest
from unittest.mock import patch

import trillim
from fastapi import APIRouter
from fastapi.testclient import TestClient

from trillim.errors import ContextOverflowError
from trillim.harnesses import DefaultHarness, SearchHarness, get_harness
from trillim.runtime import Runtime
from trillim.server import LLM, SentenceChunker, Server as ServerExport, TTS, Whisper
from trillim.server._component import Component
from trillim.server._helpers import make_id, now
from trillim.server._models import ChatCompletionRequest, CompletionRequest
from trillim.server._server import Server


class _TestComponent(Component):
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
    ):
        self._name = name
        self._calls = calls
        self._start_error = start_error
        self._stop_error = stop_error
        self.started = False

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.get(f"/{self._name}")
        async def route():
            return {"name": self._name}

        return router

    async def start(self) -> None:
        self._calls.append(f"{self._name}.start")
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    async def stop(self) -> None:
        self._calls.append(f"{self._name}.stop")
        self.started = False
        if self._stop_error is not None:
            raise self._stop_error


class _OtherTestComponent(_TestComponent):
    pass


class _ThirdTestComponent(_TestComponent):
    pass


class SdkSurfaceTests(unittest.TestCase):
    def test_trillim_lazy_exports_cover_all_public_names(self):
        self.assertIs(trillim.__getattr__("LLM"), LLM)
        self.assertIs(trillim.__getattr__("Server"), ServerExport)
        self.assertIs(trillim.__getattr__("Runtime"), Runtime)
        self.assertIs(trillim.__getattr__("TTS"), TTS)
        self.assertIs(trillim.__getattr__("SentenceChunker"), SentenceChunker)
        self.assertIs(trillim.__getattr__("Whisper"), Whisper)
        self.assertIs(trillim.__getattr__("ContextOverflowError"), ContextOverflowError)

    def test_trillim_lazy_exports_reject_unknown_names(self):
        with self.assertRaisesRegex(AttributeError, "has no attribute 'MissingThing'"):
            trillim.__getattr__("MissingThing")

    def test_make_id_prefix_and_time_helpers(self):
        with patch("trillim.server._helpers.uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "abcdef1234567890"
            self.assertEqual(make_id(), "chatcmpl-abcdef123456")

        with patch("trillim.server._helpers.time.time", return_value=123.9):
            self.assertEqual(now(), 123)

    def test_harness_registry_returns_known_harnesses_and_rejects_unknown(self):
        self.assertIs(get_harness("default"), DefaultHarness)
        self.assertIs(get_harness("search"), SearchHarness)

        with self.assertRaisesRegex(ValueError, "Unknown harness 'missing'. Available: default, search"):
            get_harness("missing")

    def test_sampling_request_validators_reject_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "temperature must be >= 0"):
            ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], temperature=-0.1)

        with self.assertRaisesRegex(ValueError, "top_p must be in \\(0, 1\\]"):
            ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], top_p=0)

        with self.assertRaisesRegex(ValueError, "top_k must be >= 1"):
            CompletionRequest(prompt="hi", top_k=0)

        with self.assertRaisesRegex(ValueError, "max_tokens must be >= 1"):
            CompletionRequest(prompt="hi", max_tokens=0)

        with self.assertRaisesRegex(ValueError, "repetition_penalty must be >= 0"):
            CompletionRequest(prompt="hi", repetition_penalty=-1)

    def test_sampling_requests_accept_valid_values(self):
        chat = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            top_p=0.5,
            top_k=2,
            max_tokens=4,
            repetition_penalty=0.1,
        )
        completion = CompletionRequest(
            prompt="hello",
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            max_tokens=1,
            repetition_penalty=0.0,
        )

        self.assertFalse(chat.stream)
        self.assertFalse(completion.stream)

    def test_server_requires_components_and_rejects_duplicate_types(self):
        with self.assertRaisesRegex(ValueError, "Server requires at least one component"):
            Server()

        with self.assertRaisesRegex(ValueError, "Duplicate component type: _TestComponent"):
            Server(_TestComponent("a", []), _TestComponent("b", []))

    def test_server_builds_app_once_and_runs_component_lifespan(self):
        calls: list[str] = []
        first = _TestComponent("alpha", calls)
        second = _OtherTestComponent("beta", calls)
        server = Server(first, second)

        self.assertIs(server.app, server.app)

        with TestClient(server.app) as client:
            self.assertTrue(first.started)
            self.assertTrue(second.started)
            self.assertEqual(client.get("/alpha").json(), {"name": "alpha"})
            self.assertEqual(client.get("/beta").json(), {"name": "beta"})

        self.assertFalse(first.started)
        self.assertFalse(second.started)
        self.assertEqual(
            calls,
            [
                "alpha.start",
                "beta.start",
                "beta.stop",
                "alpha.stop",
            ],
        )

    def test_server_rolls_back_started_components_when_lifespan_start_fails(self):
        calls: list[str] = []
        first = _TestComponent("alpha", calls)
        second = _OtherTestComponent(
            "beta",
            calls,
            stop_error=RuntimeError("beta stop failed"),
        )
        third = _ThirdTestComponent(
            "gamma",
            calls,
            start_error=RuntimeError("gamma start failed"),
        )
        server = Server(first, second, third)

        with self.assertRaisesRegex(RuntimeError, "gamma start failed"):
            with TestClient(server.app):
                pass

        self.assertFalse(first.started)
        self.assertFalse(second.started)
        self.assertFalse(third.started)
        self.assertEqual(
            calls,
            [
                "alpha.start",
                "beta.start",
                "gamma.start",
                "beta.stop",
                "alpha.stop",
            ],
        )

    def test_server_continues_shutdown_after_stop_failure(self):
        calls: list[str] = []
        first = _TestComponent("alpha", calls)
        second = _OtherTestComponent(
            "beta",
            calls,
            stop_error=RuntimeError("beta stop failed"),
        )
        server = Server(first, second)

        with self.assertRaisesRegex(RuntimeError, "beta stop failed"):
            with TestClient(server.app) as client:
                self.assertEqual(client.get("/alpha").json(), {"name": "alpha"})

        self.assertEqual(
            calls,
            [
                "alpha.start",
                "beta.start",
                "beta.stop",
                "alpha.stop",
            ],
        )

    def test_server_run_delegates_to_uvicorn(self):
        server = Server(_TestComponent("alpha", []))

        with patch("trillim.server._server.uvicorn.run") as mock_run:
            server.run(host="0.0.0.0", port=9000, reload=False)

        mock_run.assert_called_once_with(server.app, host="0.0.0.0", port=9000, reload=False)


if __name__ == "__main__":
    unittest.main()
