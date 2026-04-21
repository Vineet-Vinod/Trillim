"""Configuration and metadata types for the LLM component."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path

from trillim.components.llm._limits import DEFAULT_MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS


class LLMState(StrEnum):
    """Externally visible LLM runtime state."""

    RUNNING = "running"
    DRAINING = "draining"
    SWAPPING = "swapping"
    SERVER_ERROR = "server_error"
    UNAVAILABLE = "unavailable"


class ArchitectureType(IntEnum):
    """Numeric architecture identifiers expected by the inference worker."""

    UNKNOWN = 0
    BITNET = 1
    LLAMA = 2
    QWEN35 = 3
    BONSAI = 4
    BONSAI_TERNARY = 5


class ActivationType(IntEnum):
    """Numeric activation identifiers expected by the inference worker."""

    RELU_SQR = 0
    SILU = 1


@dataclass(frozen=True, slots=True)
class ModelRuntimeConfig:
    """Validated model metadata needed by the LLM runtime."""

    name: str
    path: Path
    arch_type: ArchitectureType
    activation: ActivationType
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    head_dim: int
    max_position_embeddings: int
    norm_eps: float
    rope_theta: float
    eos_tokens: tuple[int, ...]
    has_qkv_bias: bool
    tie_word_embeddings: bool
    has_attn_sub_norm: bool
    has_ffn_sub_norm: bool


@dataclass(frozen=True, slots=True)
class InitConfig:
    """Configured init-time options for one LLM runtime."""

    model_dir: Path
    num_threads: int = 0
    lora_dir: Path | None = None
    lora_quant: str | None = None
    unembed_quant: str | None = None


@dataclass(frozen=True, slots=True)
class SamplingDefaults:
    """Default sampling values loaded from model metadata."""

    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    rep_penalty_lookback: int = 64
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


@dataclass(frozen=True, slots=True)
class RuntimeInitInfo:
    """Public snapshot of active init-time worker options."""

    num_threads: int = 0
    lora_quant: str | None = None
    unembed_quant: str | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Public snapshot of the active LLM runtime state."""

    state: LLMState
    name: str | None
    path: str | None
    max_context_tokens: int | None
    trust_remote_code: bool
    adapter_path: str | None = None
    init_config: RuntimeInitInfo | None = None


def load_sampling_defaults(model_dir: Path) -> SamplingDefaults:
    """Load generation defaults from ``generation_config.json`` when available."""
    defaults = SamplingDefaults()
    config_path = model_dir / "generation_config.json"
    if not config_path.is_file():
        return defaults
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    max_tokens = payload.get("max_new_tokens", defaults.max_tokens)
    if isinstance(max_tokens, bool):
        max_tokens = defaults.max_tokens
    if isinstance(max_tokens, int):
        max_tokens = max(0, min(max_tokens, MAX_OUTPUT_TOKENS))
    else:
        max_tokens = defaults.max_tokens
    return SamplingDefaults(
        temperature=_float_or_default(payload.get("temperature"), defaults.temperature),
        top_k=_int_or_default(payload.get("top_k"), defaults.top_k),
        top_p=_float_or_default(payload.get("top_p"), defaults.top_p),
        repetition_penalty=_float_or_default(
            payload.get("repetition_penalty"),
            defaults.repetition_penalty,
        ),
        rep_penalty_lookback=_int_or_default(
            payload.get("rep_penalty_lookback"),
            defaults.rep_penalty_lookback,
        ),
        max_tokens=max_tokens,
    )


def _float_or_default(value, default: float) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_default(value, default: int) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        return int(value)
    except (TypeError, ValueError):
        return default
