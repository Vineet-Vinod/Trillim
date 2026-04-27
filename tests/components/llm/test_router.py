from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from trillim import _model_store
from trillim.components.llm._events import ChatUsage
from trillim.components.llm._router import (
    build_router,
    _as_http_error,
    _model_payload,
    _response_id,
    _sampling_kwargs,
    _sse,
    _usage_payload,
)
from trillim.components.llm.public import LLM
from trillim.errors import (
    AdmissionRejectedError,
    ComponentLifecycleError,
    ContextOverflowError,
    InvalidRequestError,
    ProgressTimeoutError,
    SessionClosedError,
)

from tests.support import write_llm_bundle


BONSAI_MODEL_ID = "Trillim/Bonsai-1.7B-TRNQ"
BONSAI_MODEL_DIR = _model_store.store_path_for_id(BONSAI_MODEL_ID)


class LLMRouterTests(unittest.TestCase):
    def test_router_without_lifespan_maps_not_running_chat_to_503(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_llm_bundle(root / "Local" / "model")
            with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                llm = LLM("Local/model")
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(build_router(llm, allow_hot_swap=True))
                client = TestClient(app)

                models_response = client.get("/v1/models")
                chat_response = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                swap_response = client.post(
                    "/v1/models/swap",
                    json={"model_dir": "Local/model"},
                )

        self.assertEqual(models_response.json(), {"object": "list", "data": []})
        self.assertEqual(chat_response.status_code, 503)
        self.assertIn("not running", chat_response.json()["detail"])
        self.assertEqual(swap_response.status_code, 503)

    def test_router_rejects_invalid_json_and_large_content_length(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_llm_bundle(root / "Local" / "model")
            with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                llm = LLM("Local/model")
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(build_router(llm, allow_hot_swap=False))
                client = TestClient(app)

                invalid = client.post(
                    "/v1/chat/completions",
                    content=b"{",
                    headers={"content-type": "application/json"},
                )
                too_large = client.post(
                    "/v1/chat/completions",
                    content=b"{}",
                    headers={"content-length": str(20_000_000)},
                )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(too_large.status_code, 413)

    def test_router_helper_payloads_and_error_mapping(self):
        class Request:
            temperature = 0.1
            top_k = 2
            top_p = 0.3
            repetition_penalty = 1.1
            rep_penalty_lookback = 4
            max_tokens = 5

        usage = ChatUsage(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            cached_tokens=4,
        )

        self.assertEqual(_model_payload("model"), {"id": "model", "object": "model"})
        self.assertTrue(_response_id().startswith("chatcmpl-"))
        self.assertEqual(json.loads(_sse({"a": 1}).removeprefix("data: ")), {"a": 1})
        self.assertEqual(_usage_payload(usage)["total_tokens"], 3)
        self.assertEqual(_usage_payload(None)["total_tokens"], 0)
        self.assertEqual(_sampling_kwargs(Request())["top_k"], 2)

        cases = (
            (HTTPException(status_code=418), 418),
            (InvalidRequestError("bad"), 400),
            (ContextOverflowError(2, 1), 400),
            (ComponentLifecycleError("down"), 503),
            (SessionClosedError("closed"), 409),
            (AdmissionRejectedError("busy"), 429),
            (ProgressTimeoutError("slow"), 504),
            (RuntimeError("unknown"), 503),
        )
        for exc, status_code in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(_as_http_error(exc).status_code, status_code)

    @unittest.skipUnless(
        BONSAI_MODEL_DIR.is_dir(),
        f"{BONSAI_MODEL_ID} must be installed in the Trillim model store",
    )
    def test_router_serves_real_bonsai_chat_completion_and_stream(self):
        from fastapi import FastAPI

        llm = LLM(BONSAI_MODEL_ID)
        app = FastAPI()
        app.add_event_handler("startup", llm.start)
        app.add_event_handler("shutdown", llm.stop)
        app.include_router(build_router(llm, allow_hot_swap=False))

        with TestClient(app) as client:
            models_response = client.get("/v1/models")
            completion_response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Bonsai-1.7B-TRNQ",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0,
                    "top_k": 1,
                    "max_tokens": 1,
                },
            )
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "temperature": 0,
                    "top_k": 1,
                    "max_tokens": 1,
                },
            ) as stream_response:
                stream_body = stream_response.read().decode("utf-8")

        self.assertEqual(models_response.status_code, 200)
        self.assertEqual(
            models_response.json()["data"],
            [{"id": "Bonsai-1.7B-TRNQ", "object": "model"}],
        )
        self.assertEqual(completion_response.status_code, 200)
        completion_payload = completion_response.json()
        self.assertEqual(completion_payload["model"], "Bonsai-1.7B-TRNQ")
        self.assertEqual(completion_payload["choices"][0]["finish_reason"], "stop")
        self.assertGreaterEqual(completion_payload["usage"]["completion_tokens"], 0)
        self.assertIn("chat.completion.chunk", stream_body)
        self.assertIn("data: [DONE]", stream_body)

    @unittest.skipUnless(
        BONSAI_MODEL_DIR.is_dir(),
        f"{BONSAI_MODEL_ID} must be installed in the Trillim model store",
    )
    def test_router_hot_swaps_real_bonsai_model(self):
        from fastapi import FastAPI

        llm = LLM(BONSAI_MODEL_ID, allow_hot_swap=True)
        app = FastAPI()
        app.add_event_handler("startup", llm.start)
        app.add_event_handler("shutdown", llm.stop)
        app.include_router(build_router(llm, allow_hot_swap=True))

        with TestClient(app) as client:
            response = client.post(
                "/v1/models/swap",
                json={
                    "model_dir": BONSAI_MODEL_ID,
                    "harness_name": "default",
                    "search_provider": "ddgs",
                    "search_token_budget": 4,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"object": "model.swap", "model": "Bonsai-1.7B-TRNQ"})
