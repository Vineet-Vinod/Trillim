# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""
Architecture registry and model configuration.

Defines supported architectures (BitNet, Llama, Qwen3.5) and provides
ModelConfig for extracting model dimensions from config.json.
"""

import json
import os
import struct
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional


class ArchType(IntEnum):
    """Python architecture enumeration."""

    UNKNOWN = 0
    BITNET = 1
    LLAMA = 2
    QWEN35 = 3


class ActivationType(IntEnum):
    """Activation function type."""

    RELU_SQR = 0  # ReLU squared (BitNet)
    SILU = 1  # SiLU/Swish (Llama, Qwen, Mistral)


@dataclass
class ArchInfo:
    """Architecture-specific configuration."""

    arch_type: ArchType
    activation: ActivationType
    has_attn_sub_norm: bool
    has_ffn_sub_norm: bool
    component_order: list[str]
    embedding_pattern: str = "embed_tokens"
    final_norm_pattern: str = "model.norm.weight"
    layer_pattern: str = r"\.layers\.(\d+)\."
    has_qkv_bias: bool = False


ARCH_REGISTRY: dict[str, ArchInfo] = {
    "bitnetforcausallm": ArchInfo(
        arch_type=ArchType.BITNET,
        activation=ActivationType.RELU_SQR,
        has_attn_sub_norm=True,
        has_ffn_sub_norm=True,
        component_order=[
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
        ],
    ),
    "llamaforcausallm": ArchInfo(
        arch_type=ArchType.LLAMA,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
        component_order=[
            "input_layernorm",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    ),
    "qwen3_5forconditionalgeneration": ArchInfo(
        arch_type=ArchType.QWEN35,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
        component_order=[
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
        ],
        embedding_pattern="language_model.embed_tokens",
        final_norm_pattern="model.language_model.norm.weight",
    ),
}

_VARIANT_ALIASES = {
    "codellama...": "llamaforcausallm",
    "bitnetbpeforcausallm": "bitnetforcausallm",
}
for _alias, _target in _VARIANT_ALIASES.items():
    ARCH_REGISTRY[_alias] = ARCH_REGISTRY[_target]


# LoRA target projections in the order they appear in qmodel.lora per layer
LORA_TARGETS = [
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

_ACTIVATION_MAP = {
    "silu": ActivationType.SILU,
    "swish": ActivationType.SILU,
    "relu_squared": ActivationType.RELU_SQR,
    "relu2": ActivationType.RELU_SQR,
}
_STOP_TOKEN_NAMES = (
    "<|eot_id|>",
    "<|im_end|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
)
_DEFAULT_EOS_TOKENS = {
    ArchType.BITNET: 128009,
    ArchType.LLAMA: 128009,
    ArchType.QWEN35: 248044,
}


def _extract_model_text_config(config: dict) -> dict:
    """Return the text config for text-only or multimodal checkpoints."""
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and text_config:
        normalized = dict(text_config)
        normalized["architectures"] = config.get(
            "architectures",
            text_config.get("architectures", []),
        )
        return normalized
    return config


def _get_all_tensor_names(model_dir):
    """Get all tensor names from safetensors (handles sharding)."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return list(json.load(f).get("weight_map", {}).keys())
    if os.path.exists(single_path):
        with open(single_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        return [k for k in header if k != "__metadata__"]
    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json found in {model_dir}"
    )


def _load_json_file(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_optional_json_file(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    try:
        return _load_json_file(path)
    except (json.JSONDecodeError, IOError):
        return None


def _load_tensor_names_if_available(model_dir: Optional[str]):
    if not model_dir:
        return None
    try:
        return _get_all_tensor_names(model_dir)
    except FileNotFoundError:
        return None


def _resolve_arch_info(config: dict) -> ArchInfo:
    architectures = config.get("architectures", [])
    arch_name = architectures[0] if architectures else "Unknown"
    arch_name_lower = arch_name.lower()
    if arch_name_lower not in ARCH_REGISTRY:
        raise ValueError(
            f"Unsupported architecture '{arch_name}'. Contact us and we'll add support for it."
        )
    return ARCH_REGISTRY[arch_name_lower]


def _build_bitnet_component_order(
    has_attn_sub_norm: bool,
    has_ffn_sub_norm: bool,
    attn_sub_norm_name: str,
    ffn_sub_norm_name: str,
) -> list[str]:
    component_order = ["input_layernorm"]
    if has_attn_sub_norm:
        component_order.append(attn_sub_norm_name)
    component_order.extend(
        [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
        ]
    )
    if has_ffn_sub_norm:
        component_order.append(ffn_sub_norm_name)
    component_order.append("mlp.down_proj")
    return component_order


def _resolve_bitnet_arch_info(arch_info: ArchInfo, tensor_names: Optional[list[str]]) -> ArchInfo:
    if arch_info.arch_type != ArchType.BITNET or tensor_names is None:
        return arch_info

    has_new_attn = any("attn_sub_norm" in name for name in tensor_names)
    has_old_attn = any("inner_attn_ln" in name for name in tensor_names)
    has_new_ffn = any("ffn_sub_norm" in name for name in tensor_names)
    has_old_ffn = any("ffn_layernorm" in name for name in tensor_names)

    has_attn_sub_norm = has_new_attn or has_old_attn
    has_ffn_sub_norm = has_new_ffn or has_old_ffn
    attn_sub_norm_name = (
        "self_attn.inner_attn_ln" if has_old_attn else "self_attn.attn_sub_norm"
    )
    ffn_sub_norm_name = "mlp.ffn_layernorm" if has_old_ffn else "mlp.ffn_sub_norm"

    return replace(
        arch_info,
        has_attn_sub_norm=has_attn_sub_norm,
        has_ffn_sub_norm=has_ffn_sub_norm,
        component_order=_build_bitnet_component_order(
            has_attn_sub_norm,
            has_ffn_sub_norm,
            attn_sub_norm_name,
            ffn_sub_norm_name,
        ),
    )


def _align_to_128(dim: int) -> int:
    return ((dim + 127) // 128) * 128


def _extract_dimensions(config: dict) -> dict[str, int]:
    hidden_dim_orig = config.get("hidden_size", 2560)
    intermediate_dim_orig = config.get("intermediate_size", 6912)
    num_heads = config.get("num_attention_heads", 20)
    return {
        "hidden_dim": _align_to_128(hidden_dim_orig),
        "intermediate_dim": _align_to_128(intermediate_dim_orig),
        "num_layers": config.get("num_hidden_layers", 30),
        "num_heads": num_heads,
        "num_kv_heads": config.get("num_key_value_heads", num_heads),
        "vocab_size": config.get("vocab_size", 128256),
        "head_dim": config.get("head_dim", hidden_dim_orig // num_heads),
        "max_position_embeddings": config.get("max_position_embeddings", 4096),
        "hidden_dim_orig": hidden_dim_orig,
        "intermediate_dim_orig": intermediate_dim_orig,
    }


def _resolve_activation(arch_info: ArchInfo, config: dict) -> ArchInfo:
    hidden_act = config.get("hidden_act")
    if not hidden_act:
        return arch_info

    detected_activation = _ACTIVATION_MAP.get(hidden_act.lower())
    if detected_activation is None:
        supported = list(_ACTIVATION_MAP.keys())
        raise ValueError(
            f"Unknown activation function '{hidden_act}'. Supported: {supported}"
        )
    if detected_activation == arch_info.activation:
        return arch_info
    return replace(arch_info, activation=detected_activation)


def _resolve_rope_theta(config: dict) -> float:
    rope_theta = config.get("rope_theta")
    if rope_theta is not None:
        return rope_theta

    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        nested_theta = rope_parameters.get("rope_theta")
        if nested_theta is not None:
            return nested_theta

    return 10000.0


def _resolve_qkv_bias(
    arch_info: ArchInfo,
    config: dict,
    tensor_names: Optional[list[str]],
) -> tuple[ArchInfo, bool]:
    has_qkv_bias = config.get("attention_bias", arch_info.has_qkv_bias)
    if tensor_names and any(
        "q_proj.bias" in name or "k_proj.bias" in name or "v_proj.bias" in name
        for name in tensor_names
    ):
        has_qkv_bias = True
    if has_qkv_bias == arch_info.has_qkv_bias:
        return arch_info, has_qkv_bias
    return replace(arch_info, has_qkv_bias=has_qkv_bias), has_qkv_bias


def _resolve_tied_embeddings(config: dict, tensor_names: Optional[list[str]]) -> bool:
    tie_word_embeddings = config.get("tie_word_embeddings", False)
    if tensor_names and any(name.endswith("lm_head.weight") for name in tensor_names):
        return False
    return tie_word_embeddings


def _extract_qwen35_fields(config: dict, arch_type: ArchType) -> dict[str, object]:
    if arch_type != ArchType.QWEN35:
        return {
            "layer_types": [],
            "attn_output_gate": False,
            "linear_num_key_heads": 0,
            "linear_num_value_heads": 0,
            "linear_key_head_dim": 0,
            "linear_value_head_dim": 0,
            "linear_conv_kernel_dim": 0,
        }

    return {
        "layer_types": list(config.get("layer_types", [])),
        "attn_output_gate": bool(config.get("attn_output_gate", False)),
        "linear_num_key_heads": int(config.get("linear_num_key_heads", 0)),
        "linear_num_value_heads": int(config.get("linear_num_value_heads", 0)),
        "linear_key_head_dim": int(config.get("linear_key_head_dim", 0)),
        "linear_value_head_dim": int(config.get("linear_value_head_dim", 0)),
        "linear_conv_kernel_dim": int(config.get("linear_conv_kernel_dim", 0)),
    }


def _collect_added_token_ids(
    tokenizer_data: Optional[dict],
    token_names: tuple[str, ...] = _STOP_TOKEN_NAMES,
) -> list[int]:
    if not tokenizer_data:
        return []
    token_ids = []
    for token_entry in tokenizer_data.get("added_tokens", []):
        if token_entry.get("content", "") in token_names and "id" in token_entry:
            token_ids.append(token_entry["id"])
    return token_ids


def _resolve_added_token_id(tokenizer_data: Optional[dict], token_content: str) -> Optional[int]:
    if not tokenizer_data:
        return None
    for token_entry in tokenizer_data.get("added_tokens", []):
        if token_entry.get("content") == token_content and "id" in token_entry:
            return token_entry["id"]
    return None


def _dedupe_preserving_order(values: list[int]) -> list[int]:
    seen = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _collect_eos_tokens(
    config: dict,
    arch_type: ArchType,
    model_dir: Optional[str],
    adapter_dir: Optional[str],
) -> list[int]:
    eos_tokens = []
    eot_raw = config.get("eos_token_id", _DEFAULT_EOS_TOKENS.get(arch_type, 2))
    if isinstance(eot_raw, list):
        eos_tokens.extend(eot_raw)
    else:
        eos_tokens.append(eot_raw)

    base_tokenizer_data = None
    if model_dir:
        base_tokenizer_data = _load_optional_json_file(os.path.join(model_dir, "tokenizer.json"))
        eos_tokens.extend(_collect_added_token_ids(base_tokenizer_data))

    if adapter_dir:
        lora_tokenizer_data = _load_optional_json_file(
            os.path.join(adapter_dir, "lora_tokenizer.json")
        )
        eos_tokens.extend(_collect_added_token_ids(lora_tokenizer_data))

        lora_tok_cfg = _load_optional_json_file(
            os.path.join(adapter_dir, "lora_tokenizer_config.json")
        )
        eos_token_str = lora_tok_cfg.get("eos_token") if lora_tok_cfg else None
        if eos_token_str:
            for tokenizer_data in (lora_tokenizer_data, base_tokenizer_data):
                eos_token_id = _resolve_added_token_id(tokenizer_data, eos_token_str)
                if eos_token_id is not None:
                    eos_tokens.append(eos_token_id)
                    break

    return _dedupe_preserving_order(eos_tokens)


@dataclass
class ModelConfig:
    """Extracted model configuration."""

    arch_type: ArchType
    arch_info: ArchInfo
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
    eos_tokens: list[int]
    has_qkv_bias: bool = False
    tie_word_embeddings: bool = True
    hidden_dim_orig: int = 0
    intermediate_dim_orig: int = 0
    layer_types: list[str] | None = None
    attn_output_gate: bool = False
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0

    @classmethod
    def from_config_json(
        cls, config_path: str, model_dir: Optional[str] = None,
        adapter_dir: Optional[str] = None,
    ) -> "ModelConfig":
        """Parse config.json and extract model configuration."""
        raw_config = _load_json_file(config_path)
        config = _extract_model_text_config(raw_config)
        arch_info = _resolve_arch_info(config)
        tensor_names = _load_tensor_names_if_available(model_dir)
        arch_info = _resolve_bitnet_arch_info(arch_info, tensor_names)
        dims = _extract_dimensions(config)
        norm_eps = config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-6))
        rope_theta = _resolve_rope_theta(config)
        arch_info = _resolve_activation(arch_info, config)
        arch_info, has_qkv_bias = _resolve_qkv_bias(arch_info, config, tensor_names)
        tie_word_embeddings = _resolve_tied_embeddings(config, tensor_names)
        eos_tokens = _collect_eos_tokens(config, arch_info.arch_type, model_dir, adapter_dir)
        qwen35_fields = _extract_qwen35_fields(config, arch_info.arch_type)

        return cls(
            arch_type=arch_info.arch_type,
            arch_info=arch_info,
            hidden_dim=dims["hidden_dim"],
            intermediate_dim=dims["intermediate_dim"],
            num_layers=dims["num_layers"],
            num_heads=dims["num_heads"],
            num_kv_heads=dims["num_kv_heads"],
            vocab_size=dims["vocab_size"],
            head_dim=dims["head_dim"],
            max_position_embeddings=dims["max_position_embeddings"],
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            eos_tokens=eos_tokens,
            has_qkv_bias=has_qkv_bias,
            tie_word_embeddings=tie_word_embeddings,
            hidden_dim_orig=dims["hidden_dim_orig"],
            intermediate_dim_orig=dims["intermediate_dim_orig"],
            layer_types=qwen35_fields["layer_types"],
            attn_output_gate=qwen35_fields["attn_output_gate"],
            linear_num_key_heads=qwen35_fields["linear_num_key_heads"],
            linear_num_value_heads=qwen35_fields["linear_num_value_heads"],
            linear_key_head_dim=qwen35_fields["linear_key_head_dim"],
            linear_value_head_dim=qwen35_fields["linear_value_head_dim"],
            linear_conv_kernel_dim=qwen35_fields["linear_conv_kernel_dim"],
        )
