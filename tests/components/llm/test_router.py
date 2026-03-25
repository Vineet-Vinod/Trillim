"""Tests for the LLM HTTP router."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient

from trillim import _model_store
from trillim.components.llm import ChatDoneEvent, ChatTokenEvent, ChatUsage
from trillim.components.llm import _router as llm_router
from trillim.components.llm._engine import EngineCrashedError, EngineProgressTimeoutError
from trillim.components.llm.public import LLM
from trillim.errors import AdmissionRejectedError, ProgressTimeoutError, SessionClosedError
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

    def _stream_request_model(self):
        return SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="Say hi")],
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            rep_penalty_lookback=None,
            max_tokens=None,
        )

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

    def test_stream_chat_response_skips_empty_token_events(self):
        class _LLM:
            def model_info(self):
                return SimpleNamespace(name="fake")

        class _Session:
            async def _consume_started_stream(self, _first_event, _full_text):
                yield ChatTokenEvent(text="")
                yield ChatTokenEvent(text="ok")
                yield ChatDoneEvent(text="ok", usage=ChatUsage(1, 1, 2, 0))

        async def collect() -> list[str]:
            return [
                chunk
                async for chunk in llm_router._stream_chat_response(
                    _LLM(),
                    _Session(),
                    first_event=None,
                    full_text="",
                    response_id="chatcmpl-test",
                    created=123,
                )
            ]

        chunks = asyncio.run(collect())
        body = "".join(chunks)

        self.assertIn('"delta": {"role": "assistant"}', body)
        self.assertIn('"content": "ok"', body)
        self.assertEqual(body.count('"content": "ok"'), 1)
        self.assertIn("data: [DONE]", body)

    def test_chat_completions_stream_rejects_busy_before_committing_sse(self):
        server = self._make_server(responses=["hi"])
        llm = server.components[0]

        async def reject_admission():
            raise AdmissionRejectedError("LLM is busy")

        llm._admission.acquire = reject_admission

        with TestClient(server.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"], "LLM is busy")

    def test_chat_stream_response_releases_admission_on_start_disconnect(self):
        server = self._make_server(responses=["hi"])
        llm = server.components[0]

        async def run():
            await llm.start()
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())

            async def receive():
                return {"type": "http.disconnect"}

            async def send(_message):
                raise OSError("disconnect")

            await response({"type": "http"}, receive, send)
            self.assertEqual(llm._admission.active_count, 0)
            await llm.stop()

        asyncio.run(run())

    def test_chat_stream_response_closes_session_immediately_on_disconnect(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        release = asyncio.Event()
        closed_before_release = asyncio.Event()

        class _Session:
            def __init__(self) -> None:
                self.closed = False

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                return ChatTokenEvent(text="a"), "a"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                if not release.is_set():
                    closed_before_release.set()
                self.closed = True

            async def _consume_started_stream(self, _first_event, _full_text):
                await release.wait()
                yield ChatDoneEvent(text="a", usage=ChatUsage(1, 1, 2, 0))

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()
                self._session = _Session()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                return None

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []
        receive_count = 0

        async def receive():
            nonlocal receive_count
            receive_count += 1
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.start":
                await asyncio.sleep(0)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            task = asyncio.create_task(response({"type": "http"}, receive, send))
            await asyncio.sleep(0)
            await task

        asyncio.run(run())
        self.assertEqual(messages, [])
        self.assertEqual(llm._lease.release_count, 1)
        self.assertTrue(llm._session.closed)
        self.assertTrue(closed_before_release.is_set())
        self.assertGreaterEqual(receive_count, 1)

    def test_chat_stream_response_cancels_after_receive_disconnect_and_recovers(self):
        server = self._make_server(responses=["hi"])
        llm = server.components[0]
        started = asyncio.Event()
        cancelled = asyncio.Event()
        recoveries = 0

        async def blocking_stream_events(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield ChatTokenEvent(text="done")

        async def run():
            nonlocal recoveries
            await llm.start()
            llm._harness.stream_events = blocking_stream_events
            async def recover() -> None:
                nonlocal recoveries
                recoveries += 1
            llm._recover_from_engine_failure = recover
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            messages = []

            async def receive():
                await started.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                messages.append(message)

            task = asyncio.create_task(response({"type": "http"}, receive, send))
            await asyncio.sleep(0)
            await task
            self.assertEqual(messages, [])
            self.assertEqual(llm._admission.active_count, 0)
            self.assertTrue(cancelled.is_set())
            self.assertEqual(recoveries, 1)
            await llm.stop()

        asyncio.run(run())

    def test_chat_stream_response_skips_headers_after_disconnect_before_commit(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        ready = asyncio.Event()
        closed_before_release = asyncio.Event()

        class _Session:
            def __init__(self) -> None:
                self._active_task = None

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                self._active_task = asyncio.current_task()
                ready.set()
                await asyncio.Future()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                closed_before_release.set()
                task = self._active_task
                if task is not None and not task.done() and task is not asyncio.current_task():
                    task.cancel()

            async def _consume_started_stream(self, _first_event, _full_text):
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()
                self._session = _Session()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                return None

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await ready.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            task = asyncio.create_task(response({"type": "http"}, receive, send))
            await asyncio.sleep(0)
            await task

        asyncio.run(run())
        self.assertEqual(messages, [])
        self.assertEqual(llm._lease.release_count, 1)
        self.assertTrue(closed_before_release.is_set())

    def test_chat_stream_response_drains_when_header_send_disconnects(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                return ChatTokenEvent(text="a"), "a"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

            async def _consume_started_stream(self, _first_event, _full_text):
                yield ChatDoneEvent(text="a", usage=ChatUsage(1, 1, 2, 0))

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.start":
                raise OSError("disconnect")

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_drains_when_body_send_disconnects(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                return ChatTokenEvent(text="a"), "a"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

            async def _consume_started_stream(self, _first_event, _full_text):
                yield ChatDoneEvent(text="a", usage=ChatUsage(1, 1, 2, 0))

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []
        body_sends = 0

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            nonlocal body_sends
            messages.append(message)
            if message["type"] == "http.response.body":
                body_sends += 1
                if body_sends == 1:
                    raise OSError("disconnect")

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_swallows_disconnect_before_precommit_closed_error(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        ready = asyncio.Event()
        release = asyncio.Event()

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                ready.set()
                await release.wait()
                raise SessionClosedError("ChatSession is closed")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await ready.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            task = asyncio.create_task(response({"type": "http"}, receive, send))
            await asyncio.sleep(0)
            release.set()
            await task

        asyncio.run(run())
        self.assertEqual(messages, [])
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_swallows_disconnect_before_precommit_runtime_error(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        ready = asyncio.Event()
        release = asyncio.Event()

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                ready.set()
                await release.wait()
                raise RuntimeError("boom")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await ready.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            task = asyncio.create_task(response({"type": "http"}, receive, send))
            await asyncio.sleep(0)
            release.set()
            await task

        asyncio.run(run())
        self.assertEqual(messages, [])
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_swallows_disconnect_while_sending_final_body(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

            async def _consume_started_stream(self, _first_event, _full_text):
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body" and not message["more_body"]:
                raise OSError("disconnect")

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_swallows_cancelled_stream_task(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                raise asyncio.CancelledError

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            return None

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_swallows_disconnect_while_sending_error_response(self):
        class _LLM:
            def open_session(self, _messages):
                raise AdmissionRejectedError("LLM is busy")

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            raise OSError("disconnect")

        async def run() -> None:
            response = llm_router._ChatStreamResponse(_LLM(), self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())

    def test_chat_stream_response_recovers_from_progress_timeout(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _FailingSession:
            def __init__(self) -> None:
                self.closed = False

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                raise EngineProgressTimeoutError("boom")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

        class _LLM:
            def __init__(self) -> None:
                self.recoveries = 0
                self._lease = _Lease()
                self._session = _FailingSession()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                self.recoveries += 1

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        request_model = self._stream_request_model()

        async def receive():
            await asyncio.Event().wait()

        messages = []

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, request_model)
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(messages[0]["status"], 504)
        self.assertEqual(llm._lease.release_count, 1)
        self.assertTrue(llm._session.closed)
        self.assertEqual(llm.recoveries, 1)

    def test_chat_stream_response_recovers_from_engine_crash(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _FailingSession:
            def __init__(self) -> None:
                self.closed = False

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                raise EngineCrashedError("boom")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

        class _LLM:
            def __init__(self) -> None:
                self.recoveries = 0
                self._lease = _Lease()
                self._session = _FailingSession()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                self.recoveries += 1

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        request_model = self._stream_request_model()

        async def receive():
            await asyncio.Event().wait()

        messages = []

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, request_model)
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(messages[0]["status"], 503)
        self.assertEqual(llm._lease.release_count, 1)
        self.assertTrue(llm._session.closed)
        self.assertEqual(llm.recoveries, 1)

    def test_chat_stream_response_returns_prestart_stale_error_before_sse(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                raise llm_router.SessionStaleError("ChatSession is stale")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(messages[0]["status"], 409)
        self.assertNotIn("text/event-stream", str(messages[0]["headers"]))
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_returns_prestart_closed_error_before_sse(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                raise SessionClosedError("ChatSession is closed")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(messages[0]["status"], 409)
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_returns_prestart_runtime_error_before_sse(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                raise RuntimeError("boom")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(messages[0]["status"], 503)
        self.assertEqual(llm._lease.release_count, 1)

    def test_chat_stream_response_swallows_disconnect_while_sending_prestart_json_error(self):
        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                raise RuntimeError("boom")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

        class _Lease:
            async def release(self) -> None:
                return None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            raise OSError("disconnect")

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())

    def test_chat_stream_response_recovers_from_poststart_progress_timeout(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def __init__(self) -> None:
                self.closed = False
                self._active_task = None

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                self._active_task = asyncio.current_task()
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

            async def _consume_started_stream(self, _first_event, _full_text):
                raise EngineProgressTimeoutError("boom")
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self.recoveries = 0
                self._lease = _Lease()
                self._session = _Session()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                self.recoveries += 1

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            with self.assertRaisesRegex(ProgressTimeoutError, "boom"):
                await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(llm._lease.release_count, 1)
        self.assertEqual(llm.recoveries, 1)
        self.assertTrue(llm._session.closed)

    def test_chat_stream_response_swallows_poststart_progress_timeout_after_disconnect(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def __init__(self) -> None:
                self.closed = False
                self._active_task = None

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                self._active_task = asyncio.current_task()
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

            async def _consume_started_stream(self, _first_event, _full_text):
                raise EngineProgressTimeoutError("boom")
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self.recoveries = 0
                self._lease = _Lease()
                self._session = _Session()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                self.recoveries += 1

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                raise OSError("disconnect")

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(llm._lease.release_count, 1)
        self.assertEqual(llm.recoveries, 1)
        self.assertTrue(llm._session.closed)

    def test_chat_stream_response_recovers_from_poststart_engine_crash(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def __init__(self) -> None:
                self.closed = False
                self._active_task = None

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                self._active_task = asyncio.current_task()
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

            async def _consume_started_stream(self, _first_event, _full_text):
                raise EngineCrashedError("boom")
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self.recoveries = 0
                self._lease = _Lease()
                self._session = _Session()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                self.recoveries += 1

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(llm._lease.release_count, 1)
        self.assertEqual(llm.recoveries, 1)
        self.assertTrue(llm._session.closed)

    def test_chat_stream_response_swallows_poststart_engine_crash_after_disconnect(self):
        class _Lease:
            def __init__(self) -> None:
                self.release_count = 0

            async def release(self) -> None:
                self.release_count += 1

        class _Session:
            def __init__(self) -> None:
                self.closed = False
                self._active_task = None

            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                self._active_task = asyncio.current_task()
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

            async def _consume_started_stream(self, _first_event, _full_text):
                raise EngineCrashedError("boom")
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self.recoveries = 0
                self._lease = _Lease()
                self._session = _Session()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return self._session

            async def _recover_from_engine_failure(self) -> None:
                self.recoveries += 1

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)
        messages = []

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                raise OSError("disconnect")

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            await response({"type": "http"}, receive, send)

        asyncio.run(run())
        self.assertEqual(messages[0]["type"], "http.response.start")
        self.assertEqual(llm._lease.release_count, 1)
        self.assertEqual(llm.recoveries, 1)
        self.assertTrue(llm._session.closed)

    def test_chat_stream_response_raises_closed_error_after_headers(self):
        class _Lease:
            async def release(self) -> None:
                return None

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

            async def _consume_started_stream(self, _first_event, _full_text):
                raise SessionClosedError("ChatSession is closed")
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            return None

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            with self.assertRaises(ExceptionGroup) as context:
                await response({"type": "http"}, receive, send)
            self.assertTrue(
                any(isinstance(exc, SessionClosedError) for exc in context.exception.exceptions)
            )

        asyncio.run(run())

    def test_chat_stream_response_raises_stale_error_after_headers(self):
        class _Lease:
            async def release(self) -> None:
                return None

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

            async def _consume_started_stream(self, _first_event, _full_text):
                raise llm_router.SessionStaleError("stale")
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            return None

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            with self.assertRaises(ExceptionGroup) as context:
                await response({"type": "http"}, receive, send)
            self.assertTrue(
                any(
                    isinstance(exc, llm_router.SessionStaleError)
                    for exc in context.exception.exceptions
                )
            )

        asyncio.run(run())

    def test_chat_stream_response_raises_runtime_error_after_headers(self):
        class _Lease:
            async def release(self) -> None:
                return None

        class _Session:
            def _prepare_stream_chat(self, **_kwargs):
                return "sampling"

            async def _start_prepared_stream(self, _sampling):
                return None, ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                return None

            async def close(self) -> None:
                return None

            async def _consume_started_stream(self, _first_event, _full_text):
                raise RuntimeError("boom")
                if False:  # pragma: no cover
                    yield None

        class _LLM:
            def __init__(self) -> None:
                self._lease = _Lease()

            def model_info(self):
                return SimpleNamespace(name="fake")

            def open_session(self, _messages):
                return _Session()

            class _Admission:
                def __init__(self, lease) -> None:
                    self._lease = lease

                async def acquire(self):
                    return self._lease

        llm = _LLM()
        llm._admission = llm._Admission(llm._lease)

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            return None

        async def run() -> None:
            response = llm_router._ChatStreamResponse(llm, self._stream_request_model())
            with self.assertRaises(ExceptionGroup) as context:
                await response({"type": "http"}, receive, send)
            self.assertTrue(
                any(isinstance(exc, RuntimeError) for exc in context.exception.exceptions)
            )

        asyncio.run(run())

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
