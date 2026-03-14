# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Shared Pydantic sampling parameter schemas for HTTP and engine calls."""

from pydantic import BaseModel, ValidationError, field_validator


class CommonSamplingParams(BaseModel):
    """Sampling parameters shared across HTTP and direct engine callers."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("temperature must be >= 0")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float | None) -> float | None:
        if v is not None and not (0 < v <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        return v

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("top_k must be >= 1")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def _check_repetition_penalty(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("repetition_penalty must be >= 0")
        return v


class HttpSamplingParams(CommonSamplingParams):
    """HTTP sampling parameters aligned with the OpenAI-compatible request schema."""

    max_tokens: int | None = None

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_tokens must be >= 1")
        return v


class EngineSamplingParams(CommonSamplingParams):
    """Engine sampling parameters, including engine-only knobs."""

    max_tokens: int | None = None
    rep_penalty_lookback: int | None = None

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("max_tokens must be >= 0")
        return v


def first_validation_error(exc: ValidationError) -> str:
    """Return the first user-facing validation message from a Pydantic error."""

    errors = exc.errors(include_url=False)
    if not errors:
        return str(exc)
    msg = str(errors[0]["msg"])
    prefix = "Value error, "
    if msg.startswith(prefix):
        return msg[len(prefix):]
    return msg
