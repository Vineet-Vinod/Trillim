"""Tests for server/app composition."""

import unittest

from fastapi import APIRouter
from fastapi.testclient import TestClient

from trillim.components import Component
from trillim.server import Server


class _RouteComponent(Component):
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self._component_name = name

    @property
    def component_name(self) -> str:
        return self._component_name

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.get(f"/{self.component_name}")
        async def route():
            return {"name": self.component_name}

        return router

    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")

    async def stop(self) -> None:
        self.calls.append(f"{self.component_name}.stop")


class _BrokenStartComponent(_RouteComponent):
    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")
        raise RuntimeError("broken start")


class ServerTests(unittest.TestCase):
    def test_server_requires_components(self):
        with self.assertRaisesRegex(ValueError, "at least one component"):
            Server()

    def test_server_rejects_duplicate_component_names(self):
        calls: list[str] = []
        with self.assertRaisesRegex(ValueError, "Duplicate component name"):
            Server(_RouteComponent(calls, "dup"), _RouteComponent(calls, "dup"))

    def test_server_builds_app_and_runs_lifecycle(self):
        calls: list[str] = []
        server = Server(
            _RouteComponent(calls, "one"),
            _RouteComponent(calls, "two"),
            allow_llm_hot_swap=True,
        )
        self.assertTrue(server.allow_llm_hot_swap)
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
            self.assertEqual(client.get("/one").json(), {"name": "one"})
            self.assertEqual(client.get("/two").json(), {"name": "two"})
        self.assertEqual(
            calls,
            ["one.start", "two.start", "two.stop", "one.stop"],
        )

    def test_server_stops_started_components_when_startup_fails(self):
        calls: list[str] = []
        server = Server(_RouteComponent(calls, "ok"), _BrokenStartComponent(calls, "bad"))
        with self.assertRaisesRegex(RuntimeError, "Component startup failed"):
            with TestClient(server.app):
                pass
        self.assertEqual(calls, ["ok.start", "bad.start", "ok.stop"])

