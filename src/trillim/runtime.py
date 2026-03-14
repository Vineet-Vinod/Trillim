# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Application runtime for embedding Trillim components synchronously."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import AsyncIterator, Iterator

from trillim.server._component import Component


class _SyncAsyncIterator(Iterator):
    """Drive an async iterator from the runtime's loop thread."""

    def __init__(self, runtime: Runtime, iterator: AsyncIterator):
        self._runtime = runtime
        self._iterator = iterator
        self._closed = False

    def __iter__(self) -> _SyncAsyncIterator:
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            return self._runtime._submit_coroutine(self._iterator.__anext__()).result()
        except StopAsyncIteration as exc:
            self._closed = True
            raise StopIteration from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._runtime._submit_coroutine(self._iterator.aclose()).result()
        except RuntimeError:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _RuntimeManagedProxy:
    """Expose a runtime-managed target through synchronous wrappers."""

    def __init__(self, runtime: Runtime, managed):
        self._runtime = runtime
        self._managed = managed

    def __getattr__(self, name: str):
        attr = self._runtime._get_managed_attr(self._managed, name)
        if callable(attr):
            def _call(*args, **kwargs):
                result = self._runtime._invoke_managed_attr(
                    self._managed, name, args, kwargs
                )
                return self._runtime._syncify_result(result)

            return _call
        return self._runtime._syncify_result(attr)


class _RuntimeComponentProxy(_RuntimeManagedProxy):
    """Expose a runtime-owned component as the synchronous service entry point."""


class _RuntimeObjectProxy(_RuntimeManagedProxy):
    """Expose returned runtime-managed handles, such as sessions, synchronously."""

    def __iter__(self):
        if not hasattr(self._managed, "__aiter__"):
            raise TypeError(f"{type(self._managed).__name__!r} is not iterable")
        try:
            iterator = self._runtime._invoke_managed_attr(
                self._managed,
                "__aiter__",
                (),
                {},
            )
        except AttributeError as exc:
            raise TypeError(f"{type(self._managed).__name__!r} is not iterable") from exc
        return _SyncAsyncIterator(self._runtime, iterator)


class Runtime:
    """Own a background event loop and expose components synchronously."""

    def __init__(self, *components: Component):
        if not components:
            raise ValueError("Runtime requires at least one component")

        seen: set[type] = set()
        for component in components:
            if type(component) in seen:
                raise ValueError(
                    f"Duplicate component type: {type(component).__name__}"
                )
            seen.add(type(component))

        self._components = tuple(components)
        self._component_names = {
            type(component).__name__.lower(): component for component in self._components
        }
        self._component_proxies = {
            name: _RuntimeComponentProxy(self, component)
            for name, component in self._component_names.items()
        }
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._lock = threading.Lock()
        self._started = False

    @property
    def components(self) -> tuple[Component, ...]:
        return self._components

    @property
    def started(self) -> bool:
        return self._started

    def __getattr__(self, name: str):
        if name in self._component_proxies:
            return self._component_proxies[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._component_proxies))

    def __enter__(self) -> Runtime:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> Runtime:
        with self._lock:
            if self._started:
                return self

            self._loop_ready.clear()
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="trillim-runtime",
                daemon=True,
            )
            self._thread.start()
            self._loop_ready.wait()

            try:
                self._submit_to_loop(self._start_components()).result()
            except Exception:
                self._shutdown_loop()
                raise

            self._started = True
            return self

    def stop(self) -> None:
        with self._lock:
            if self._loop is None or self._thread is None:
                self._started = False
                return

            error: Exception | None = None
            try:
                self._submit_to_loop(self._stop_components()).result()
            except Exception as exc:
                error = exc
            finally:
                self._shutdown_loop()
                self._started = False

            if error is not None:
                raise error

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

        pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()

    async def _start_components(self) -> None:
        started: list[Component] = []
        try:
            for component in self._components:
                await component.start()
                started.append(component)
        except Exception:
            for component in reversed(started):
                try:
                    await component.stop()
                except Exception:
                    pass
            raise

    async def _stop_components(self) -> None:
        first_error: Exception | None = None
        for component in reversed(self._components):
            try:
                await component.stop()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _shutdown_loop(self) -> None:
        if self._loop is None or self._thread is None:
            self._loop = None
            self._thread = None
            return

        loop = self._loop
        thread = self._thread
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        self._loop = None
        self._thread = None
        self._loop_ready.clear()

    def _submit_to_loop(self, coro):
        if self._loop is None or self._thread is None:
            raise RuntimeError("Runtime not started")
        loop = self._loop
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _submit_coroutine(self, coro):
        if not self._started:
            raise RuntimeError("Runtime not started")
        return self._submit_to_loop(coro)

    async def _get_attr_async(self, obj, name: str):
        return getattr(obj, name)

    def _get_managed_attr(self, obj, name: str):
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            raise RuntimeError("Runtime not started")
        return asyncio.run_coroutine_threadsafe(
            self._get_attr_async(obj, name), loop
        ).result()

    async def _invoke_attr_async(
        self,
        obj,
        name: str,
        args: tuple,
        kwargs: dict,
    ):
        attr = getattr(obj, name)
        if not callable(attr):
            if args or kwargs:
                raise TypeError(f"{name!r} is not callable")
            return attr
        result = attr(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _invoke_managed_attr(
        self,
        obj,
        name: str,
        args: tuple,
        kwargs: dict,
    ):
        if not self._started:
            raise RuntimeError("Runtime not started")
        return self._submit_to_loop(
            self._invoke_attr_async(obj, name, args, kwargs)
        ).result()

    def _syncify_result(self, result):
        if inspect.isawaitable(result):
            result = self._submit_coroutine(result).result()
        if hasattr(result, "__anext__"):
            return _SyncAsyncIterator(self, result)
        # Results marked as runtime-managed handles stay distinct from
        # top-level components so object proxies can layer extra behavior such
        # as sync iteration for async-iterable sessions.
        if getattr(result, "_runtime_proxy", False):
            return _RuntimeObjectProxy(self, result)
        return result
