"""FastAPI application construction for composed Trillim components."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Iterable

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from trillim.components import Component
from trillim.errors import ComponentLifecycleError


async def _stop_components(components: Iterable[Component]) -> None:
    first_error: Exception | None = None
    for component in reversed(tuple(components)):
        try:
            await component.stop()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise ComponentLifecycleError("Component shutdown failed") from first_error


def build_app(components: Iterable[Component]) -> FastAPI:
    """Build a FastAPI app from a set of components."""
    items = tuple(components)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started: list[Component] = []
        try:
            for component in items:
                await component.start()
                started.append(component)
        except Exception as exc:
            try:
                await _stop_components(started)
            except Exception:
                pass
            raise ComponentLifecycleError("Component startup failed") from exc
        try:
            yield
        finally:
            await _stop_components(started)

    app = FastAPI(title="Trillim API", version="0.7.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        unhealthy_components: dict[str, dict[str, object]] = {}
        for component in items:
            detail = _component_unhealthy_detail(component)
            if detail is not None:
                unhealthy_components[component.component_name] = detail
        if unhealthy_components:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "components": unhealthy_components,
                },
            )
        return {"status": "ok"}

    for component in items:
        app.include_router(component.router())

    return app


def _component_unhealthy_detail(component: Component) -> dict[str, object] | None:
    model_info = getattr(component, "model_info", None)
    if not callable(model_info):
        return None
    info = model_info()
    state = getattr(info, "state", None)
    state_value = getattr(state, "value", state)
    if state_value in {None, "running"}:
        return None
    return {"state": state_value}
