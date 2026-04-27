"""Private search harness for models that emit ``<search>...</search>`` tags."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from trillim.components.llm._events import ChatEvent, ChatFinalTextEvent, ChatTokenEvent
from trillim.components.llm._incremental_decode import IncrementalDecoder
from trillim.components.llm._session import _ChatSession
from trillim.harnesses._base import _Harness
from trillim.harnesses.search.client import SearchClient
from trillim.harnesses.search.metrics import SearchMetrics
from trillim.harnesses.search.provider import (
    FALLBACK_SEARCH_FAILURE_MESSAGE,
    MAX_SEARCH_ITERATIONS,
    SearchAuthenticationError,
    SearchError,
    extract_search_query,
)


class _SearchHarness(_Harness):
    """Run a bounded search loop before streaming the final answer."""

    def __init__(
        self,
        llm,
        runtime,
        *,
        search_provider: str,
        search_token_budget: int,
        _search_client_factory=SearchClient,
    ) -> None:
        super().__init__(llm, runtime)
        self._search = _search_client_factory(
            provider_name=search_provider,
            token_budget=search_token_budget,
        )
        self._search_token_budget = search_token_budget

    async def stream_events(
        self,
        session: _ChatSession,
        **sampling: Any,
    ) -> AsyncIterator[ChatEvent]:
        """Run buffered search iterations, then stream the final answer."""
        self._reset_usage()
        metrics = SearchMetrics()
        for _ in range(MAX_SEARCH_ITERATIONS - 1):
            cached_tokens = session.cached_token_count
            token_ids = session._prepare_generation(messages=session._messages)
            full_text, completion_token_ids = await self._generate_buffered(
                session,
                token_ids,
                **sampling,
            )
            query = extract_search_query(full_text)
            metrics.record_generation(
                prompt_tokens=len(token_ids) - cached_tokens,
                completion_tokens=len(completion_token_ids),
            )
            if query is None:
                self._apply_metrics(metrics)
                session._messages.append({"role": "assistant", "content": full_text})
                session._pending_token_ids = (*token_ids, *completion_token_ids)
                if full_text:
                    yield ChatTokenEvent(text=full_text)
                yield ChatFinalTextEvent(text=full_text)
                return
            try:
                search_content = await self._search.search(query)
            except SearchAuthenticationError:
                self._apply_metrics(metrics)
                raise
            except SearchError:
                search_content = FALLBACK_SEARCH_FAILURE_MESSAGE
            if search_content != FALLBACK_SEARCH_FAILURE_MESSAGE:
                search_content = self._trim_search_content(search_content)
            session._messages.append({"role": "assistant", "content": full_text})
            session._cached_token_ids = (*token_ids, *completion_token_ids)
            session._messages_in_kv = len(session._messages)
            session._messages.append({"role": "search", "content": search_content})

        cached_tokens = session.cached_token_count
        token_ids = session._prepare_generation(messages=session._messages)
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        completion_token_ids: list[int] = []
        token_stream = self._generate_tokens(session, token_ids, **sampling)
        try:
            async for token_id in token_stream:
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
        metrics.record_generation(
            prompt_tokens=len(token_ids) - cached_tokens,
            completion_tokens=len(completion_token_ids),
        )
        self._apply_metrics(metrics)
        session._messages.append({"role": "assistant", "content": full_text})
        session._pending_token_ids = (*token_ids, *completion_token_ids)
        yield ChatFinalTextEvent(text=full_text)

    async def _generate_buffered(
        self,
        session: _ChatSession,
        token_ids: list[int],
        **sampling: Any,
    ) -> tuple[str, list[int]]:
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        completion_token_ids: list[int] = []
        token_stream = self._generate_tokens(session, token_ids, **sampling)
        try:
            async for token_id in token_stream:
                completion_token_ids.append(token_id)
                full_text += decoder.decode(token_id)
        finally:
            await token_stream.aclose()
        full_text += decoder.flush()
        return full_text, completion_token_ids

    def _trim_search_content(self, content: str) -> str:
        token_ids = list(
            self.tokenizer.encode(
                content,
                add_special_tokens=False,
            )
        )
        if len(token_ids) <= self._search_token_budget:
            return content
        return self.tokenizer.decode(
            token_ids[: self._search_token_budget],
            skip_special_tokens=True,
        ).strip()

    def _apply_metrics(self, metrics: SearchMetrics) -> None:
        self._prompt_tokens = metrics.prompt_tokens
        self._completion_tokens = metrics.completion_tokens
