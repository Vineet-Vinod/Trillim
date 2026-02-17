# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Pydantic request/response models for the Trillim API."""

import enum

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Chat / Completion models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class _SamplingValidators:
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

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_tokens must be >= 1")
        return v

    @field_validator("repetition_penalty")
    @classmethod
    def _check_repetition_penalty(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("repetition_penalty must be >= 0")
        return v


class ChatCompletionRequest(_SamplingValidators, BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    stream: bool = False


class CompletionRequest(_SamplingValidators, BaseModel):
    model: str = ""
    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    stream: bool = False


class ChatChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: ChatChoiceDelta | None = None
    finish_reason: str | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ServerState(enum.Enum):
    RUNNING = "running"
    SWAPPING = "swapping"
    NO_MODEL = "no_model"


class LoadModelRequest(BaseModel):
    model_dir: str
    adapter_dir: str | None = None
    lora: bool | None = None  # None = auto (true if adapter_dir provided)
    threads: int | None = None  # None = keep server default, 0 = auto


class LoadModelResponse(BaseModel):
    status: str
    model: str
    recompiled: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Voice / Audio models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = ""
    input: str
    voice: str | None = None
    response_format: str = "wav"  # "wav" or "pcm"


class TranscriptionResponse(BaseModel):
    text: str


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    type: str  # "predefined" or "custom"


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]


class VoiceCreateResponse(BaseModel):
    voice_id: str
    status: str
