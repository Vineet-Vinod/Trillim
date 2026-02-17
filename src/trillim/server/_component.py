# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Component ABC for composable server pieces."""

import abc

from fastapi import APIRouter


class Component(abc.ABC):
    @abc.abstractmethod
    def router(self) -> APIRouter: ...

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...
