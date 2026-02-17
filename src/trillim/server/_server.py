# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Server compositor — composes components into a FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from ._component import Component


class Server:
    """Composes Component instances into a single FastAPI application.

    Usage::

        Server(LLM("models/BitNet")).run()
        Server(LLM("models/BitNet"), Whisper(), TTS()).run()
        Server(TTS()).run()
    """

    def __init__(self, *components: Component):
        if not components:
            raise ValueError("Server requires at least one component")

        # Reject duplicate component types
        seen: set[type] = set()
        for c in components:
            if type(c) in seen:
                raise ValueError(f"Duplicate component type: {type(c).__name__}")
            seen.add(type(c))

        self._components = components
        self._app: FastAPI | None = None

    @property
    def app(self) -> FastAPI:
        """Expose the FastAPI instance for custom routes."""
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self) -> FastAPI:
        components = self._components

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            for c in components:
                await c.start()
            yield
            for c in reversed(components):
                await c.stop()

        app = FastAPI(title="Trillim API", version="0.1.0", lifespan=lifespan)

        for c in components:
            app.include_router(c.router())

        return app

    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs) -> None:
        """Start the server with uvicorn."""
        uvicorn.run(self.app, host=host, port=port, **kwargs)
