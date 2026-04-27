"""Internal search harness usage bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SearchMetrics:
    """Final-turn usage bookkeeping for a search-orchestrated turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record_generation(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Store authoritative usage for the final request."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
