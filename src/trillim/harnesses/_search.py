# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""SearchHarness — multi-step harness for models that emit <search> XML tags."""

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from trillim.engine import InferenceEngine
from trillim.token_utils import IncrementalDecoder
from ._base import Harness
from ._search_utils import SearchClient, SearchError, extract_search_query


class SearchHarness(Harness):
    """Harness for models fine-tuned to emit <search>query</search> tags."""

    MAX_SEARCH_ITERATIONS = 3
    DEBUG: ClassVar[bool] = False

    def __init__(self, engine: InferenceEngine, search_provider: str = "ddgs"):
        super().__init__(engine)
        self._search = SearchClient(provider_name=search_provider)

    async def _generate_buffered(self, messages: list[dict], **sampling: Any) -> tuple[str, int]:
        """Generate a full response non-streaming."""
        token_ids, prompt_str = self._prepare_tokens(messages)
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        completion_tokens = 0
        async for token_id in self.engine.generate(
            token_ids=token_ids, prompt_str=prompt_str, **sampling,
        ):
            completion_tokens += 1
            full_text += decoder.decode(token_id)
        return full_text, completion_tokens

    async def run(self, messages: list[dict], **sampling: Any) -> AsyncIterator[str]:
        """Orchestration loop: intermediate steps buffered, final step streamed.

        Yields sentinels ([Searching: ...], [Synthesizing...]) before model
        text so the CLI can distinguish them from the actual response.
        """
        self._last_completion_tokens = 0
        yield "[Spin-Jump-Spinning...]\n"
        for i in range(self.MAX_SEARCH_ITERATIONS - 1):
            full_text, completion_tokens = await self._generate_buffered(messages, **sampling)

            if self.DEBUG:
                yield f"\n--- Step {i + 1} ---\n{full_text}\n"

            query = extract_search_query(full_text)
            if query is None:
                # No search tag — yield buffered text as final response
                self._last_completion_tokens = completion_tokens
                yield full_text
                messages.append({"role": "assistant", "content": full_text})
                self._update_cache(messages)
                return

            # Yield sentinel BEFORE searching so the user sees it while waiting
            yield f"[Searching: {query}]\n"

            # Keep cache aligned with model-generated state only.
            messages.append({"role": "assistant", "content": full_text})
            self._update_cache(messages)

            try:
                results = await self._search.search(query)
            except SearchError:
                yield "[Search unavailable]\n"
                messages.append({"role": "search", "content": "Search unavailable, please answer from your knowledge."})
                break

            messages.append({"role": "search", "content": results})

            if self.DEBUG:
                yield f"[Search results]\n{results}\n"
                
            yield "[Synthesizing...]\n"

        # Final iteration: stream token-by-token
        token_ids, prompt_str = self._prepare_tokens(messages)
        decoder = IncrementalDecoder(self.tokenizer)
        full_text = ""
        async for token_id in self.engine.generate(
            token_ids=token_ids, prompt_str=prompt_str, **sampling,
        ):
            self._last_completion_tokens += 1
            chunk = decoder.decode(token_id)
            full_text += chunk
            yield chunk

        messages.append({"role": "assistant", "content": full_text})
        self._update_cache(messages)
