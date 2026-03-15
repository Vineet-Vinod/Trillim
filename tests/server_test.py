# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""End-to-end HTTP tests for the composed Trillim server."""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
import urllib.request
from contextlib import contextmanager

import uvicorn
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from trillim.server._component import Component
from trillim.server._server import Server


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json_get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def _bytes_get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


class _StatusComponent(Component):
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls
        self.started = False

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.get(f"/{self._name}")
        async def route():
            return {"name": self._name, "started": self.started}

        return router

    async def start(self) -> None:
        self._calls.append(f"{self._name}.start")
        self.started = True

    async def stop(self) -> None:
        self._calls.append(f"{self._name}.stop")
        self.started = False


class _MetricsComponent(_StatusComponent):
    pass


class _StreamComponent(Component):
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.started = False

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/stream")
        async def route():
            def _iter():
                yield b"alpha\n"
                yield b"beta\n"

            return StreamingResponse(_iter(), media_type="text/plain")

        return router

    async def start(self) -> None:
        self._calls.append("stream.start")
        self.started = True

    async def stop(self) -> None:
        self._calls.append("stream.stop")
        self.started = False


@contextmanager
def _running_server(*components: Component):
    port = _find_free_port()
    server = Server(*components)
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(
            server.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            lifespan="on",
        )
    )
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/{components[0]._name}", timeout=0.2):
                break
        except Exception:
            time.sleep(0.05)
    else:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Timed out waiting for test server to start")

    try:
        yield base_url
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)


class ServerE2ETests(unittest.TestCase):
    def test_server_serves_multiple_component_routes_over_real_http(self):
        calls: list[str] = []
        status = _StatusComponent("status", calls)
        metrics = _MetricsComponent("metrics", calls)

        with _running_server(status, metrics) as base_url:
            status_code, status_body = _json_get(f"{base_url}/status")
            metrics_code, metrics_body = _json_get(f"{base_url}/metrics")

        self.assertEqual(status_code, 200)
        self.assertEqual(status_body, {"name": "status", "started": True})
        self.assertEqual(metrics_code, 200)
        self.assertEqual(metrics_body, {"name": "metrics", "started": True})
        self.assertFalse(status.started)
        self.assertFalse(metrics.started)
        self.assertEqual(
            calls,
            [
                "status.start",
                "metrics.start",
                "metrics.stop",
                "status.stop",
            ],
        )

    def test_server_streaming_route_works_over_real_http(self):
        calls: list[str] = []
        status = _StatusComponent("status", calls)
        stream = _StreamComponent(calls)

        with _running_server(status, stream) as base_url:
            status_code, payload = _bytes_get(f"{base_url}/stream")

        self.assertEqual(status_code, 200)
        self.assertEqual(payload, b"alpha\nbeta\n")
        self.assertEqual(
            calls,
            [
                "status.start",
                "stream.start",
                "stream.stop",
                "status.stop",
            ],
        )


if __name__ == "__main__":
    unittest.main()
