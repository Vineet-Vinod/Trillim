"""Default harness for direct single-pass LLM generation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from trillim.components.llm._events import ChatEvent, ChatFinalTextEvent, ChatTokenEvent
from trillim.components.llm._incremental_decode import IncrementalDecoder
from trillim.harnesses.base import Harness


class DefaultHarness(Harness):
    """Run one direct generation with no tool orchestration."""

    async def stream_events(
        self,
        session,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Stream token and final-text events for one generation."""
        self._completion_tokens = 0
        token_ids = session._prepare_generation()
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        async for token_id in self._engine.generate(token_ids=token_ids, **sampling):
            self._completion_tokens += 1
            chunk = decoder.decode(token_id)
            if not chunk:
                continue
            full_text += chunk
            yield ChatTokenEvent(text=chunk)
        session._commit_assistant_turn(full_text)
        yield ChatFinalTextEvent(text=full_text)
