# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Harness ABC — abstract base for inference harnesses that steer multi-step execution."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from trillim.engine import InferenceEngine
from trillim.events import ChatEvent, ChatTokenEvent

if TYPE_CHECKING:
    from trillim.server._llm import ChatSession


class Harness(abc.ABC):
    """Abstract base for inference harnesses that steer multi-step execution.

    Subclasses implement stream_events() for full orchestration.
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self._last_completion_tokens = 0

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def arch_config(self):
        return self.engine.arch_config

    async def run(self, session: ChatSession, **sampling: Any) -> AsyncIterator[str]:
        """Compatibility text stream built from structured chat events."""
        async for event in self.stream_events(session, **sampling):
            if isinstance(event, ChatTokenEvent):
                yield event.text

    @abc.abstractmethod
    async def stream_events(
        self,
        session: ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Structured orchestration loop for app-facing streaming APIs."""
        ...
        yield  # type: ignore  # abstract async generator
