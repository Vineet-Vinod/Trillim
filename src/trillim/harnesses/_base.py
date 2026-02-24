# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Harness ABC — abstract base for inference harnesses that steer multi-step execution."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from trillim.engine import InferenceEngine


@dataclass
class StepResult:
    """Result of a single harness step."""
    text: str                    # what the model generated this step
    messages: list[dict]         # updated message list (may include tool results)
    done: bool                   # True = no more steps needed


class Harness(abc.ABC):
    """Abstract base for inference harnesses that steer multi-step execution.

    Subclasses implement step() for one iteration and run() for full orchestration.

    Set DEBUG = True in source to print all intermediate generations.
    When False, intermediate steps emit only short sentinels.
    """

    DEBUG: ClassVar[bool] = False

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def arch_config(self):
        return self.engine.arch_config

    @abc.abstractmethod
    async def step(self, messages: list[dict], **sampling: Any) -> StepResult:
        """One generation + optional tool execution (non-streaming).

        Returns a StepResult with generated text, updated messages, and
        whether execution is complete.
        """
        ...

    @abc.abstractmethod
    async def run(self, messages: list[dict], **sampling: Any) -> AsyncIterator[str]:
        """Full orchestration loop. Yields text chunks to display.

        Calls step() in a loop for multi-step harnesses. Streams the final
        generation token-by-token. Updates messages in place (appends the
        final assistant response). Updates engine._cached_prompt_str for
        KV cache reuse on the next turn.
        """
        ...
        yield  # type: ignore  # abstract async generator

    def _update_cache(self, messages: list[dict]) -> None:
        """Update the engine's cached_prompt_str for KV cache reuse."""
        tokenizer = self.tokenizer
        has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        if has_template:
            self.engine._cached_prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

    def _prepare_tokens(self, messages: list[dict]) -> tuple[list[int], str | None]:
        """Render messages via chat template and encode to token IDs.

        Returns (token_ids, prompt_str). prompt_str is for string-level KV
        cache matching, or None if no chat template.
        """
        tokenizer = self.tokenizer
        has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        if has_template:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            return token_ids, prompt_str
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"
            token_ids = tokenizer.encode(prompt)
            return token_ids, None
