"""Source-model parsing for local quantization."""

from __future__ import annotations

import json
import re
import struct
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

from trillim._bundle_metadata import canonicalize_model_config
from trillim.components.llm._config import ArchitectureType

LORA_TARGETS = (
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

_ACTIVATION_MAP = {
    "relu_squared": "relu_squared",
    "relu2": "relu_squared",
    "relu_sqr": "relu_squared",
    "silu": "silu",
    "swish": "silu",
}


@dataclass(frozen=True, slots=True)
class _ArchInfo:
    arch_type: ArchitectureType
    component_order: tuple[str, ...]
    embedding_key: str
    final_norm_key: str
    layer_pattern: str = r"\.layers\.(\d+)\."


_ARCH_REGISTRY: dict[str, _ArchInfo] = {
    "bitnetforcausallm": _ArchInfo(
        arch_type=ArchitectureType.BITNET,
        component_order=(
            "input_layernorm",
            "self_attn.attn_sub_norm",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.ffn_sub_norm",
            "mlp.down_proj",
        ),
        embedding_key="model.embed_tokens.weight",
        final_norm_key="model.norm.weight",
    ),
    "llamaforcausallm": _ArchInfo(
        arch_type=ArchitectureType.LLAMA,
        component_order=(
            "input_layernorm",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        embedding_key="model.embed_tokens.weight",
        final_norm_key="model.norm.weight",
    ),
    "qwen3_5forconditionalgeneration": _ArchInfo(
        arch_type=ArchitectureType.QWEN35,
        component_order=(
            "input_layernorm",
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        embedding_key="model.language_model.embed_tokens.weight",
        final_norm_key="model.language_model.norm.weight",
    ),
    "bitnetbpeforcausallm": _ArchInfo(
        arch_type=ArchitectureType.BITNET,
        component_order=(
            "input_layernorm",
            "self_attn.attn_sub_norm",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.ffn_sub_norm",
            "mlp.down_proj",
        ),
        embedding_key="model.embed_tokens.weight",
        final_norm_key="model.norm.weight",
    ),
    "qwen3forcausallm": _ArchInfo(
        arch_type=ArchitectureType.BONSAI,
        component_order=(
            "input_layernorm",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        embedding_key="model.embed_tokens.weight",
        final_norm_key="model.norm.weight",
    ),
}


@dataclass(frozen=True, slots=True)
class ModelQuantizeConfig:
    arch_type: ArchitectureType
    arch_name: str
    arch_info: _ArchInfo
    hidden_dim: int
    intermediate_dim: int
    hidden_dim_orig: int
    intermediate_dim_orig: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    head_dim: int
    max_position_embeddings: int
    norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    yarn_factor: float | None
    original_max_position_embeddings: int | None
    yarn_beta_slow: float | None
    yarn_beta_fast: float | None
    tie_word_embeddings: bool
    source_model: str


def load_model_config(model_dir: Path) -> ModelQuantizeConfig:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Model config must be a JSON object in {config_path}")
    config = canonicalize_model_config(raw)
    arch_info = _resolve_arch_info(config)
    arch_info = _resolve_bonsai_arch_info(model_dir, arch_info)
    tensor_names = _load_tensor_names_if_available(model_dir)
    arch_info = _resolve_bitnet_arch_info(arch_info, tensor_names)
    dimensions = _extract_dimensions(config)
    rope_theta = _resolve_rope_theta(config)
    partial_rotary_factor = _resolve_partial_rotary_factor(config)
    (
        yarn_factor,
        original_max_position_embeddings,
        yarn_beta_slow,
        yarn_beta_fast,
    ) = _resolve_yarn_scaling(config)
    _resolve_activation(config)
    tie_word_embeddings = _resolve_tied_embeddings(config, tensor_names)
    return ModelQuantizeConfig(
        arch_type=arch_info.arch_type,
        arch_name=arch_info.arch_type.name.lower(),
        arch_info=arch_info,
        hidden_dim=dimensions["hidden_dim"],
        intermediate_dim=dimensions["intermediate_dim"],
        hidden_dim_orig=dimensions["hidden_dim_orig"],
        intermediate_dim_orig=dimensions["intermediate_dim_orig"],
        num_layers=dimensions["num_layers"],
        num_heads=dimensions["num_heads"],
        num_kv_heads=dimensions["num_kv_heads"],
        vocab_size=dimensions["vocab_size"],
        head_dim=dimensions["head_dim"],
        max_position_embeddings=dimensions["max_position_embeddings"],
        norm_eps=float(config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-6))),
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        yarn_factor=yarn_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        yarn_beta_slow=yarn_beta_slow,
        yarn_beta_fast=yarn_beta_fast,
        tie_word_embeddings=tie_word_embeddings,
        source_model=str(raw.get("_name_or_path", "")),
    )


def _resolve_arch_info(config: dict) -> _ArchInfo:
    architectures = config.get("architectures", [])
    arch_name = architectures[0] if architectures else "unknown"
    try:
        return _ARCH_REGISTRY[arch_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture '{arch_name}'") from exc


def _resolve_bitnet_arch_info(
    arch_info: _ArchInfo,
    tensor_names: list[str] | None,
) -> _ArchInfo:
    if arch_info.arch_type != ArchitectureType.BITNET or not tensor_names:
        return arch_info
    has_old_attn = any("inner_attn_ln" in name for name in tensor_names)
    has_old_ffn = any("ffn_layernorm" in name for name in tensor_names)
    if not has_old_attn and not has_old_ffn:
        return arch_info
    component_order = list(arch_info.component_order)
    if has_old_attn:
        component_order[1] = "self_attn.inner_attn_ln"
    if has_old_ffn:
        component_order[-2] = "mlp.ffn_layernorm"
    return replace(arch_info, component_order=tuple(component_order))


def _resolve_bonsai_arch_info(model_dir: Path, arch_info: _ArchInfo) -> _ArchInfo:
    if arch_info.arch_type != ArchitectureType.BONSAI:
        return arch_info
    if _readme_indicates_bonsai_ternary(model_dir):
        return replace(arch_info, arch_type=ArchitectureType.BONSAI_TERNARY)
    return arch_info


def _extract_dimensions(config: dict) -> dict[str, int]:
    hidden_dim_orig = _require_positive_int(config.get("hidden_size"), "hidden_size")
    intermediate_dim_orig = _require_positive_int(
        config.get("intermediate_size"),
        "intermediate_size",
    )
    num_heads = _require_positive_int(
        config.get("num_attention_heads"),
        "num_attention_heads",
    )
    return {
        "hidden_dim": _align_to_128(hidden_dim_orig),
        "intermediate_dim": _align_to_128(intermediate_dim_orig),
        "hidden_dim_orig": hidden_dim_orig,
        "intermediate_dim_orig": intermediate_dim_orig,
        "num_layers": _require_positive_int(config.get("num_hidden_layers"), "num_hidden_layers"),
        "num_heads": num_heads,
        "num_kv_heads": _require_positive_int(
            config.get("num_key_value_heads", num_heads),
            "num_key_value_heads",
        ),
        "vocab_size": _require_positive_int(config.get("vocab_size"), "vocab_size"),
        "head_dim": _require_positive_int(
            config.get("head_dim", hidden_dim_orig // num_heads),
            "head_dim",
        ),
        "max_position_embeddings": _require_positive_int(
            config.get("max_position_embeddings", 4096),
            "max_position_embeddings",
        ),
    }


def _resolve_rope_theta(config: dict) -> float:
    rope_theta = config.get("rope_theta")
    if rope_theta is None:
        rope_parameters = config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        else:
            rope_theta = 10000.0
    return float(rope_theta)


def _resolve_partial_rotary_factor(config: dict) -> float:
    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict) and rope_parameters.get("partial_rotary_factor") is not None:
        return float(rope_parameters["partial_rotary_factor"])
    factor = config.get("partial_rotary_factor", 1.0)
    return float(factor)


def _resolve_yarn_scaling(
    config: dict,
) -> tuple[float | None, int | None, float | None, float | None]:
    rope_scaling = config.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
        return None, None, None, None
    if rope_scaling.get("rope_type") != "yarn":
        return None, None, None, None
    factor = rope_scaling.get("factor")
    original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings")
    beta_slow = rope_scaling.get("beta_slow")
    beta_fast = rope_scaling.get("beta_fast")
    if factor is None or original_max_position_embeddings is None:
        raise ValueError("YaRN rope_scaling requires factor and original_max_position_embeddings")
    return (
        float(factor),
        _require_positive_int(
            original_max_position_embeddings,
            "rope_scaling.original_max_position_embeddings",
        ),
        None if beta_slow is None else float(beta_slow),
        None if beta_fast is None else float(beta_fast),
    )


def _resolve_activation(config: dict) -> str:
    hidden_act = config.get("hidden_act")
    if hidden_act is None:
        return "silu"
    try:
        return _ACTIVATION_MAP[str(hidden_act).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation function: {hidden_act}") from exc


def _resolve_tied_embeddings(config: dict, tensor_names: list[str] | None) -> bool:
    del tensor_names
    return bool(config.get("tie_word_embeddings", False))


def _load_tensor_names_if_available(model_dir: Path) -> list[str] | None:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            return [str(name) for name in weight_map]
    if single_path.is_file():
        with single_path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        return [str(name) for name in header if name != "__metadata__"]
    return None


def _readme_indicates_bonsai_ternary(model_dir: Path) -> bool:
    readme_path = model_dir / "README.md"
    if not readme_path.is_file():
        warnings.warn(
            (
                f"Could not find {readme_path}; defaulting Qwen3ForCausalLM Bonsai "
                "detection to binary. Grouped-ternary models may be misidentified."
            ),
            stacklevel=3,
        )
        return False
    try:
        content = readme_path.read_text(encoding="utf-8").lower()
    except OSError:
        return False
    has_ternary = "ternary" in content
    has_one_bit = any(token in content for token in ("1-bit", "1 bit", "1bit"))
    return has_ternary and not has_one_bit


def _align_to_128(value: int) -> int:
    return ((value + 127) // 128) * 128


def _require_positive_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


def layer_index_for_key(key: str, arch_info: _ArchInfo) -> int | None:
    match = re.search(arch_info.layer_pattern, key)
    if match is None:
        return None
    return int(match.group(1))
