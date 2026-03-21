"""Base harness protocol for LLM orchestration."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from trillim.components.llm._events import ChatEvent, ChatTokenEvent

if TYPE_CHECKING:
    from trillim.components.llm._engine import InferenceEngine
    from trillim.components.llm._session import ChatSession


class Harness(abc.ABC):
    """Abstract base class for LLM orchestration harnesses."""

    def __init__(self, engine: InferenceEngine) -> None:
        """Bind the harness to an engine."""
        self._engine = engine
        self._completion_tokens = 0

    @property
    def completion_tokens(self) -> int:
        """Return the number of completion tokens emitted in the last turn."""
        return self._completion_tokens

    @property
    def tokenizer(self):
        """Return the engine tokenizer."""
        return self._engine.tokenizer

    async def stream_text(
        self,
        session: ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[str]:
        """Yield only text fragments from structured harness events."""
        async for event in self.stream_events(session, **sampling):
            if isinstance(event, ChatTokenEvent):
                yield event.text

    @abc.abstractmethod
    async def stream_events(
        self,
        session: ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Yield structured chat events for a single assistant turn."""
        yield  # pragma: no cover
