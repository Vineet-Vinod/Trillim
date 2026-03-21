"""Model directory validation and metadata extraction for LLMs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from trillim.components.llm._config import (
    ActivationType,
    ArchitectureType,
    ModelRuntimeConfig,
)
from trillim.errors import ModelValidationError

_STOP_TOKEN_NAMES = (
    "<|eot_id|>",
    "<|im_end|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
)
_DEFAULT_EOS_TOKENS = {
    ArchitectureType.BITNET: 128009,
    ArchitectureType.LLAMA: 128009,
    ArchitectureType.QWEN35: 248044,
}


@dataclass(frozen=True, slots=True)
class _ArchitectureInfo:
    arch_type: ArchitectureType
    activation: ActivationType
    has_attn_sub_norm: bool
    has_ffn_sub_norm: bool
    has_qkv_bias: bool = False


_ARCH_REGISTRY: dict[str, _ArchitectureInfo] = {
    "bitnetforcausallm": _ArchitectureInfo(
        arch_type=ArchitectureType.BITNET,
        activation=ActivationType.RELU_SQR,
        has_attn_sub_norm=True,
        has_ffn_sub_norm=True,
    ),
    "llamaforcausallm": _ArchitectureInfo(
        arch_type=ArchitectureType.LLAMA,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    ),
    "qwen3_5forconditionalgeneration": _ArchitectureInfo(
        arch_type=ArchitectureType.QWEN35,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    ),
    "bitnetbpeforcausallm": _ArchitectureInfo(
        arch_type=ArchitectureType.BITNET,
        activation=ActivationType.RELU_SQR,
        has_attn_sub_norm=True,
        has_ffn_sub_norm=True,
    ),
}
_ACTIVATION_MAP = {
    "relu_squared": ActivationType.RELU_SQR,
    "relu2": ActivationType.RELU_SQR,
    "relu_sqr": ActivationType.RELU_SQR,
    "silu": ActivationType.SILU,
    "swish": ActivationType.SILU,
}


def validate_model_dir(model_dir: str | Path) -> ModelRuntimeConfig:
    """Validate a model directory and extract runtime metadata."""
    path = Path(model_dir).expanduser()
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelValidationError(f"Model directory does not exist: {model_dir}") from exc
    if not path.is_dir():
        raise ModelValidationError(f"Model directory is not a directory: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ModelValidationError(f"config.json not found in {path}")
    _require_runtime_artifacts(path)
    config = _load_json(config_path)
    config = _extract_text_config(config)
    arch_info = _resolve_arch_info(config)
    dimensions = _extract_dimensions(config)
    eos_tokens = _collect_eos_tokens(config, arch_info.arch_type, path)
    return ModelRuntimeConfig(
        name=path.name,
        path=path,
        arch_type=arch_info.arch_type,
        activation=_resolve_activation(config, arch_info),
        hidden_dim=dimensions["hidden_dim"],
        intermediate_dim=dimensions["intermediate_dim"],
        num_layers=dimensions["num_layers"],
        num_heads=dimensions["num_heads"],
        num_kv_heads=dimensions["num_kv_heads"],
        vocab_size=dimensions["vocab_size"],
        head_dim=dimensions["head_dim"],
        max_position_embeddings=dimensions["max_position_embeddings"],
        norm_eps=float(config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-6))),
        rope_theta=_resolve_rope_theta(config),
        eos_tokens=tuple(eos_tokens),
        has_qkv_bias=_resolve_qkv_bias(config, arch_info),
        tie_word_embeddings=_resolve_tied_embeddings(config),
        has_attn_sub_norm=arch_info.has_attn_sub_norm,
        has_ffn_sub_norm=arch_info.has_ffn_sub_norm,
    )


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"Could not read JSON from {path}") from exc


def _extract_text_config(config: dict) -> dict:
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and text_config:
        normalized = dict(text_config)
        normalized["architectures"] = config.get(
            "architectures",
            text_config.get("architectures", []),
        )
        return normalized
    return config


def _resolve_arch_info(config: dict) -> _ArchitectureInfo:
    architectures = config.get("architectures", [])
    arch_name = architectures[0] if architectures else "unknown"
    try:
        return _ARCH_REGISTRY[arch_name.lower()]
    except KeyError as exc:
        raise ModelValidationError(
            f"Unsupported model architecture: {arch_name}"
        ) from exc


def _require_runtime_artifacts(model_dir: Path) -> None:
    for filename in ("qmodel.tensors", "rope.cache"):
        if not (model_dir / filename).is_file():
            raise ModelValidationError(f"{filename} not found in {model_dir}")


def _extract_dimensions(config: dict) -> dict[str, int]:
    hidden_dim = _require_positive_int(config.get("hidden_size"), "hidden_size")
    intermediate_dim = _require_positive_int(
        config.get("intermediate_size"),
        "intermediate_size",
    )
    num_heads = _require_positive_int(
        config.get("num_attention_heads"),
        "num_attention_heads",
    )
    num_kv_heads = _require_positive_int(
        config.get("num_key_value_heads", num_heads),
        "num_key_value_heads",
    )
    max_position_embeddings = _require_positive_int(
        config.get("max_position_embeddings", 4096),
        "max_position_embeddings",
    )
    vocab_size = _require_positive_int(config.get("vocab_size"), "vocab_size")
    head_dim = int(config.get("head_dim", hidden_dim // num_heads))
    if head_dim <= 0:
        raise ModelValidationError("head_dim must be a positive integer")
    return {
        "hidden_dim": _align_to_128(hidden_dim),
        "intermediate_dim": _align_to_128(intermediate_dim),
        "num_layers": _require_positive_int(config.get("num_hidden_layers"), "num_hidden_layers"),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
    }


def _align_to_128(value: int) -> int:
    return ((value + 127) // 128) * 128


def _require_positive_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ModelValidationError(f"{field_name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ModelValidationError(f"{field_name} must be a positive integer")
    return number


def _resolve_activation(config: dict, arch_info: _ArchitectureInfo) -> ActivationType:
    hidden_act = config.get("hidden_act")
    if hidden_act is None:
        return arch_info.activation
    try:
        return _ACTIVATION_MAP[str(hidden_act).lower()]
    except KeyError as exc:
        raise ModelValidationError(
            f"Unsupported activation function: {hidden_act}"
        ) from exc


def _resolve_rope_theta(config: dict) -> float:
    rope_theta = config.get("rope_theta")
    if rope_theta is None:
        rope_parameters = config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        else:
            rope_theta = 10000.0
    try:
        return float(rope_theta)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("rope_theta must be numeric") from exc


def _resolve_qkv_bias(
    config: dict,
    arch_info: _ArchitectureInfo,
) -> bool:
    return bool(config.get("attention_bias", arch_info.has_qkv_bias))


def _resolve_tied_embeddings(config: dict) -> bool:
    return bool(config.get("tie_word_embeddings", False))


def _collect_eos_tokens(
    config: dict,
    arch_type: ArchitectureType,
    model_dir: Path,
) -> list[int]:
    eos_raw = config.get("eos_token_id", _DEFAULT_EOS_TOKENS.get(arch_type, 2))
    if isinstance(eos_raw, list):
        eos_tokens = [int(token_id) for token_id in eos_raw]
    else:
        eos_tokens = [int(eos_raw)]
    tokenizer_payload = _load_optional_json(model_dir / "tokenizer.json")
    eos_tokens.extend(_collect_added_tokens(tokenizer_payload))
    deduped: list[int] = []
    seen: set[int] = set()
    for token_id in eos_tokens:
        if token_id in seen:
            continue
        seen.add(token_id)
        deduped.append(token_id)
    if not deduped:
        raise ModelValidationError("No EOS tokens could be determined for the model")
    return deduped


def _load_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_added_tokens(tokenizer_payload: dict | None) -> list[int]:
    if not tokenizer_payload:
        return []
    token_ids: list[int] = []
    for token in tokenizer_payload.get("added_tokens", []):
        if token.get("content") in _STOP_TOKEN_NAMES and "id" in token:
            token_ids.append(int(token["id"]))
    return token_ids
