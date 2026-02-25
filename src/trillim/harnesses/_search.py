# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""SearchHarness — multi-step harness for models that emit <search> XML tags."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from trillim.engine import InferenceEngine
from trillim.token_utils import IncrementalDecoder
from ._base import Harness, StepResult
from ._search_utils import DuckDuckGoSearch, SearchError, extract_search_query


class SearchHarness(Harness):
    """Harness for models fine-tuned to emit <search>query</search> tags."""

    MAX_SEARCH_ITERATIONS = 3
    DEBUG: ClassVar[bool] = False

    def __init__(self, engine: InferenceEngine):
        super().__init__(engine)
        self._search = DuckDuckGoSearch()

    async def _generate_buffered(self, messages: list[dict], **sampling: Any) -> str:
        """Generate a full response non-streaming."""
        token_ids, prompt_str = self._prepare_tokens(messages)
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        async for token_id in self.engine.generate(
            token_ids=token_ids, prompt_str=prompt_str, **sampling,
        ):
            full_text += decoder.decode(token_id)
        return full_text

    async def step(self, messages: list[dict], **sampling: Any) -> StepResult:
        """Generate one response and execute search if requested.

        Raises SearchError if search is needed but fails.
        Does not modify messages if SearchError is raised.
        """
        full_text = await self._generate_buffered(messages, **sampling)

        query = extract_search_query(full_text)
        if query is None:
            messages.append({"role": "assistant", "content": full_text})
            self._update_cache(messages)
            return StepResult(text=full_text, messages=messages, done=True)

        # May raise SearchError — messages untouched in that case
        results = await self._search.search(query)
        messages.append({"role": "assistant", "content": full_text})
        messages.append({"role": "search", "content": results})
        self._update_cache(messages)
        return StepResult(text=full_text, messages=messages, done=False)

    async def run(self, messages: list[dict], **sampling: Any) -> AsyncIterator[str]:
        """Orchestration loop: intermediate steps buffered, final step streamed.

        Yields sentinels ([Searching: ...], [Synthesizing...]) before model
        text so the CLI can distinguish them from the actual response.
        """
        for i in range(self.MAX_SEARCH_ITERATIONS - 1):
            full_text = await self._generate_buffered(messages, **sampling)

            if self.DEBUG:
                yield f"\n--- Step {i + 1} ---\n{full_text}\n"

            query = extract_search_query(full_text)
            if query is None:
                # No search tag — yield buffered text as final response
                yield full_text
                messages.append({"role": "assistant", "content": full_text})
                self._update_cache(messages)
                return

            # Yield sentinel BEFORE searching so the user sees it while waiting
            yield f"[Searching: {query}]\n"

            try:
                results = await self._search.search(query)
            except SearchError:
                yield "[Search unavailable]\n"
                break

            messages.append({"role": "assistant", "content": full_text})
            messages.append({"role": "search", "content": results})
            self._update_cache(messages)

            if self.DEBUG:
                yield f"[Search results]\n{results}\n"

        # Final iteration: stream token-by-token
        yield "[Synthesizing...]\n"

        token_ids, prompt_str = self._prepare_tokens(messages)
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        async for token_id in self.engine.generate(
            token_ids=token_ids, prompt_str=prompt_str, **sampling,
        ):
            chunk = decoder.decode(token_id)
            full_text += chunk
            yield chunk

        messages.append({"role": "assistant", "content": full_text})
        self._update_cache(messages)
