"""Structured chat event types for the LLM component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(slots=True, frozen=True)
class ChatUsage:
    """Usage accounting for a completed chat turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int


@dataclass(slots=True, frozen=True)
class ChatTokenEvent:
    """Incremental token text emitted during generation."""

    text: str
    type: Literal["token"] = "token"


@dataclass(slots=True, frozen=True)
class ChatFinalTextEvent:
    """Final normalized assistant text for a turn."""

    text: str
    type: Literal["final_text"] = "final_text"


@dataclass(slots=True, frozen=True)
class ChatDoneEvent:
    """Terminal event for a completed chat turn."""

    text: str
    usage: ChatUsage
    type: Literal["done"] = "done"


ChatEvent: TypeAlias = ChatTokenEvent | ChatFinalTextEvent | ChatDoneEvent
