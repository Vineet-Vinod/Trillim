"""Tests for the runtime sync facade."""

from __future__ import annotations

import asyncio
import unittest

from trillim.components import Component
from trillim.runtime import Runtime, _RuntimeObjectProxy


class _Session:
    _runtime_proxy = True

    def __init__(self, calls: list):
        self.calls = calls
        self.state = "ready"

    async def ping(self) -> str:
        self.calls.append("session.ping")
        return "pong"

    async def __aiter__(self):
        self.calls.append("session.__aiter__")
        yield b"a"
        yield b"b"


class _EchoComponent(Component):
    def __init__(self, calls: list[str], name: str = "echo") -> None:
        self.calls = calls
        self._component_name = name
        self.started = False

    @property
    def component_name(self) -> str:
        return self._component_name

    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")
        self.started = True

    async def stop(self) -> None:
        self.calls.append(f"{self.component_name}.stop")
        self.started = False

    async def ping(self, value: str) -> str:
        self.calls.append(f"{self.component_name}.ping:{value}")
        return value.upper()

    async def stream(self):
        self.calls.append(f"{self.component_name}.stream")
        yield "alpha"
        yield "beta"

    def session(self):
        self.calls.append(f"{self.component_name}.session")
        return _Session(self.calls)


class _BrokenComponent(_EchoComponent):
    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")
        raise RuntimeError("boom")


class RuntimeTests(unittest.TestCase):
    def test_runtime_requires_components(self):
        with self.assertRaisesRegex(ValueError, "at least one component"):
            Runtime()

    def test_runtime_rejects_duplicate_component_names(self):
        calls: list[str] = []
        with self.assertRaisesRegex(ValueError, "Duplicate component name"):
            Runtime(_EchoComponent(calls), _EchoComponent(calls))

    def test_runtime_starts_and_stops_in_order(self):
        calls: list[str] = []
        with Runtime(_EchoComponent(calls, "one"), _EchoComponent(calls, "two")) as runtime:
            self.assertTrue(runtime.started)
        self.assertEqual(
            calls,
            ["one.start", "two.start", "two.stop", "one.stop"],
        )

    def test_runtime_syncifies_async_methods_iterators_and_runtime_objects(self):
        calls: list[str] = []
        with Runtime(_EchoComponent(calls)) as runtime:
            self.assertEqual(runtime.echo.ping("hello"), "HELLO")
            self.assertEqual(list(runtime.echo.stream()), ["alpha", "beta"])
            session = runtime.echo.session()
            self.assertIsInstance(session, _RuntimeObjectProxy)
            self.assertEqual(session.ping(), "pong")
            self.assertEqual(list(session), [b"a", b"b"])
        self.assertIn("echo.session", calls)
        self.assertIn("session.ping", calls)

    def test_runtime_stops_started_components_if_startup_fails(self):
        calls: list[str] = []
        runtime = Runtime(_EchoComponent(calls, "one"), _BrokenComponent(calls, "two"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            runtime.start()
        self.assertEqual(calls, ["one.start", "two.start", "one.stop"])
        self.assertFalse(runtime.started)

