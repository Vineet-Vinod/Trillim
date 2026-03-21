"""Shared component base types and exports."""

from __future__ import annotations

from fastapi import APIRouter


class Component:
    """Composable application component with async lifecycle hooks."""

    @property
    def component_name(self) -> str:
        """Return the attribute name used by the runtime facade."""
        return type(self).__name__.lower()

    def router(self) -> APIRouter:
        """Return the component's API router."""
        return APIRouter()

    async def start(self) -> None:
        """Start the component."""

    async def stop(self) -> None:
        """Stop the component."""

