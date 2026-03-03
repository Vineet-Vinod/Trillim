# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""DefaultHarness — passthrough harness with no tool use."""

from collections.abc import AsyncIterator
from typing import Any

from trillim.token_utils import IncrementalDecoder
from ._base import Harness


class DefaultHarness(Harness):
    """Passthrough harness — single generation, no tool calls."""

    async def run(self, messages: list[dict], **sampling: Any) -> AsyncIterator[str]:
        """Stream tokens directly from the engine."""
        self._last_completion_tokens = 0
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
