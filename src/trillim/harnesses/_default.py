"""Private default harness for direct single-pass LLM generation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from trillim.components.llm._events import ChatEvent, ChatFinalTextEvent, ChatTokenEvent
from trillim.components.llm._incremental_decode import IncrementalDecoder
from trillim.components.llm._session import _ChatSession
from trillim.harnesses._base import _Harness


class _DefaultHarness(_Harness):
    """Run one direct generation with no tool orchestration."""

    async def stream_events(
        self,
        session: _ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Stream token and final-text events for one generation."""
        self._reset_usage()
        cached_tokens = session.cached_token_count
        token_ids = session._prepare_generation(messages=session._messages)
        self._prompt_tokens = len(token_ids) - cached_tokens
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        completion_token_ids: list[int] = []
        token_stream = self._generate_tokens(session, token_ids, **sampling)
        try:
            async for token_id in token_stream:
                self._completion_tokens += 1
                completion_token_ids.append(token_id)
                chunk = decoder.decode(token_id)
                if not chunk:
                    continue
                full_text += chunk
                yield ChatTokenEvent(text=chunk)
        finally:
            await token_stream.aclose()
        chunk = decoder.flush()
        if chunk:
            full_text += chunk
            yield ChatTokenEvent(text=chunk)
        session._messages.append({"role": "assistant", "content": full_text})
        session._pending_token_ids = (*token_ids, *completion_token_ids)
        yield ChatFinalTextEvent(text=full_text)
