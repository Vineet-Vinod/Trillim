"""Validation models and helpers for LLM inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from trillim.components.llm._limits import (
    MAX_MESSAGES,
    MAX_MESSAGE_CHARS,
    MAX_MODEL_NAME_CHARS,
    MAX_MODEL_PATH_CHARS,
    MAX_OUTPUT_TOKENS,
    TOTAL_MESSAGE_TEXT_LIMIT_BYTES,
)
from trillim.harnesses.search.provider import normalize_provider_name, validate_harness_name
from trillim.errors import InvalidRequestError


class SamplingOptions(BaseModel):
    """Validated sampling settings for one generation."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=200)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)

    def to_kwargs(self) -> dict[str, float | int | None]:
        """Convert the model to plain kwargs."""
        return self.model_dump()


class ChatMessageInput(BaseModel):
    """Validated external chat message input."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "search"]
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value:
            raise ValueError("message content must not be empty")
        if len(value) > MAX_MESSAGE_CHARS:
            raise ValueError(
                f"message content exceeds the {MAX_MESSAGE_CHARS} character limit"
            )
        return value


class ChatRequestInput(SamplingOptions):
    """Validated HTTP chat-completions request."""

    messages: tuple[ChatMessageInput, ...]
    model: str | None = Field(default=None, max_length=MAX_MODEL_NAME_CHARS)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: tuple[ChatMessageInput, ...]) -> tuple[ChatMessageInput, ...]:
        if not value:
            raise ValueError("messages must not be empty")
        if len(value) > MAX_MESSAGES:
            raise ValueError(f"messages exceed the limit of {MAX_MESSAGES}")
        return value


class SwapModelRequestInput(BaseModel):
    """Validated HTTP hot-swap request."""

    model_config = ConfigDict(extra="forbid")

    model_dir: str = Field(min_length=1, max_length=MAX_MODEL_PATH_CHARS)
    harness_name: str | None = Field(default=None, max_length=MAX_MODEL_NAME_CHARS)
    search_provider: str | None = Field(default=None, max_length=MAX_MODEL_NAME_CHARS)
    search_token_budget: int | None = Field(default=None, ge=1)


def validate_chat_request(
    payload: object,
    *,
    active_model_name: str | None,
) -> ChatRequestInput:
    """Validate a chat-completions request payload."""
    request = _validate_model(ChatRequestInput, payload)
    validate_messages(
        request.messages,
        require_user_turn=True,
        allow_empty=False,
    )
    if request.model is not None and request.model != active_model_name:
        raise InvalidRequestError(
            f"Requested model {request.model!r} does not match the active model"
        )
    return request


def validate_swap_request(payload: object) -> SwapModelRequestInput:
    """Validate a hot-swap request payload."""
    request = _validate_model(SwapModelRequestInput, payload)
    try:
        if request.harness_name is not None:
            validate_harness_name(request.harness_name)
        if request.search_provider is not None:
            normalize_provider_name(request.search_provider)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    return request


def validate_sampling_options(**kwargs) -> SamplingOptions:
    """Validate SDK sampling kwargs."""
    return _validate_model(SamplingOptions, kwargs)


def validate_messages(
    messages: Sequence[ChatMessageInput | dict[str, str]],
    *,
    require_user_turn: bool,
    allow_empty: bool,
) -> tuple[ChatMessageInput, ...]:
    """Validate a message sequence for SDK session use."""
    if not allow_empty and not messages:
        raise InvalidRequestError("messages must not be empty")
    converted = tuple(
        message
        if isinstance(message, ChatMessageInput)
        else _validate_model(ChatMessageInput, message)
        for message in messages
    )
    if len(converted) > MAX_MESSAGES:
        raise InvalidRequestError(f"messages exceed the limit of {MAX_MESSAGES}")
    total_bytes = sum(len(message.content.encode("utf-8")) for message in converted)
    if total_bytes > TOTAL_MESSAGE_TEXT_LIMIT_BYTES:
        raise InvalidRequestError(
            "messages exceed the total text budget before tokenization"
        )
    if require_user_turn and converted and converted[-1].role == "assistant":
        raise InvalidRequestError(
            "the last message already contains an assistant reply; add a new user or system message before chatting again"
        )
    return converted


def _validate_model(model_type: type[BaseModel], payload: object):
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else "invalid request"
        raise InvalidRequestError(message) from exc
