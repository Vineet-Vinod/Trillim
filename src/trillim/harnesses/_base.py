"""Private base harness abstraction for LLM orchestration."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from trillim.components.llm._events import ChatEvent

if TYPE_CHECKING:
    from trillim.components.llm.public import LLM, _RuntimeSnapshot
    from trillim.components.llm._session import _ChatSession


class _Harness(abc.ABC):
    """Internal base class for session-owned LLM harness implementations."""

    def __init__(self, llm: LLM, runtime: _RuntimeSnapshot) -> None:
        self._llm = llm
        self._runtime = runtime
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def prompt_tokens(self) -> int:
        """Return the prompt-token count for the last completed turn."""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Return the completion-token count for the last completed turn."""
        return self._completion_tokens

    @property
    def tokenizer(self):
        """Return the runtime tokenizer."""
        return self._runtime.tokenizer

    def _reset_usage(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0

    async def _generate_tokens(
        self,
        session: _ChatSession,
        token_ids: Sequence[int],
        **sampling: Any,
    ) -> AsyncIterator[int]:
        stream = self._llm._generate(
            self._runtime,
            session._runtime_epoch,
            token_ids=token_ids,
            **sampling,
        )
        try:
            async for token_id in stream:
                yield token_id
        finally:
            await stream.aclose()

    @abc.abstractmethod
    async def stream_events(
        self,
        session: _ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Yield structured chat events for a single assistant turn."""
        yield  # pragma: no cover
