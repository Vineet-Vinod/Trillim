# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""SearchHarness — multi-step harness for models that emit <search> XML tags."""

from collections.abc import AsyncIterator
from typing import Any

from trillim.engine import InferenceEngine
from trillim.events import (
    ChatEvent,
    ChatFinalTextEvent,
    ChatSearchResultEvent,
    ChatSearchStartedEvent,
    ChatTokenEvent,
)
from trillim.token_utils import IncrementalDecoder
from ._base import Harness
from ._search_utils import SearchClient, SearchError, extract_search_query


class SearchHarness(Harness):
    """Harness for models fine-tuned to emit <search>query</search> tags."""

    MAX_SEARCH_ITERATIONS = 3

    def __init__(self, engine: InferenceEngine, search_provider: str = "ddgs"):
        super().__init__(engine)
        self._search = SearchClient(provider_name=search_provider)

    async def _generate_buffered(self, session, **sampling: Any) -> tuple[str, list[int]]:
        """Generate a full response non-streaming."""
        token_ids, prompt_str = session._prepare_reply()
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        generated_token_ids: list[int] = []
        async for token_id in self.engine.generate(
            token_ids=token_ids, prompt_str=prompt_str, **sampling,
        ):
            generated_token_ids.append(token_id)
            full_text += decoder.decode(token_id)
        return full_text, generated_token_ids

    async def stream_events(
        self,
        session,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Structured orchestration loop with explicit search and token events."""
        self._last_completion_tokens = 0
        for _ in range(self.MAX_SEARCH_ITERATIONS - 1):
            full_text, generated_token_ids = await self._generate_buffered(
                session,
                **sampling,
            )

            query = extract_search_query(full_text)
            if query is None:
                # No search tag — yield buffered text as final response
                self._last_completion_tokens = len(generated_token_ids)
                yield ChatTokenEvent(text=full_text)
                session._finalize_assistant(full_text, generated_token_ids)
                yield ChatFinalTextEvent(text=full_text)
                return

            yield ChatSearchStartedEvent(query=query)

            # Keep cache aligned with model-generated state only.
            session._finalize_assistant(full_text, generated_token_ids)

            try:
                results = await self._search.search(query)
            except SearchError:
                results = "Search unavailable, please answer from your knowledge."
                yield ChatSearchResultEvent(
                    query=query,
                    content=results,
                    available=False,
                )
                session._append_message("search", results)
                break

            session._append_message("search", results)
            yield ChatSearchResultEvent(query=query, content=results)

        # Final iteration: stream token-by-token
        token_ids, prompt_str = session._prepare_reply()
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        generated_token_ids: list[int] = []
        async for token_id in self.engine.generate(
            token_ids=token_ids, prompt_str=prompt_str, **sampling,
        ):
            self._last_completion_tokens += 1
            generated_token_ids.append(token_id)
            chunk = decoder.decode(token_id)
            full_text += chunk
            yield ChatTokenEvent(text=chunk)

        session._finalize_assistant(full_text, generated_token_ids)
        yield ChatFinalTextEvent(text=full_text)
