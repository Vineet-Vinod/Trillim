"""Server wrapper for composed Trillim components."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from trillim._app import build_app
from trillim.components import Component


class Server:
    """Compose Trillim components into a FastAPI server."""

    def __init__(self, *components: Component):
        """Create a server from one or more components."""
        if not components:
            raise ValueError("Server requires at least one component")
        seen_names: set[str] = set()
        for component in components:
            if component.component_name in seen_names:
                raise ValueError(
                    f"Duplicate component name: {component.component_name}"
                )
            seen_names.add(component.component_name)
        self._components = tuple(components)
        self._app: FastAPI | None = None

    @property
    def components(self) -> tuple[Component, ...]:
        """Return the server's composed components."""
        return self._components

    @property
    def app(self) -> FastAPI:
        """Return the lazily constructed FastAPI application."""
        if self._app is None:
            self._app = build_app(self._components)
        return self._app

    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs) -> None:
        """Run the composed FastAPI application with uvicorn."""
        uvicorn.run(self.app, host=host, port=port, **kwargs)
