# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""
Architecture registry and model configuration.

Defines supported architectures (BitNet, Llama, Qwen2, Mistral) and provides
ModelConfig for extracting model dimensions from config.json.
"""

import json
import os
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class ArchType(IntEnum):
    """Architecture enumeration - matches C++ ArchType enum."""

    UNKNOWN = 0
    BITNET = 1
    LLAMA = 2


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

# Binary file format constants
MAGIC_TENSORS = b"TRLM"
MAGIC_ROPE    = b"TRRC"
MAGIC_LORA    = b"TRLA"
FORMAT_VERSION_TENSORS = 2
FORMAT_VERSION_ROPE    = 1
FORMAT_VERSION_LORA    = 2


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

    @classmethod
    def from_config_json(
        cls, config_path: str, model_dir: Optional[str] = None,
        adapter_dir: Optional[str] = None,
    ) -> "ModelConfig":
        """Parse config.json and extract model configuration."""
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Detect architecture (case-insensitive lookup)
        architectures = config.get("architectures", [])
        arch_name = architectures[0] if architectures else "Unknown"
        arch_name_lower = arch_name.lower()

        if arch_name_lower not in ARCH_REGISTRY:
            raise ValueError(
                f"Unsupported architecture '{arch_name}'. Contact us and we'll add support for it."
            )

        arch_info = ARCH_REGISTRY[arch_name_lower]

        # Try to detect sub-norms from actual model tensors if model_dir provided
        has_attn_sub_norm = arch_info.has_attn_sub_norm
        has_ffn_sub_norm = arch_info.has_ffn_sub_norm
        attn_sub_norm_name = "self_attn.attn_sub_norm"
        ffn_sub_norm_name = "mlp.ffn_sub_norm"

        if model_dir and arch_info.arch_type == ArchType.BITNET:
            try:
                tensor_names = _get_all_tensor_names(model_dir)

                # Check for sub-norms with different naming conventions
                # Newer: attn_sub_norm / ffn_sub_norm
                # Older: inner_attn_ln / ffn_layernorm
                has_new_attn = any("attn_sub_norm" in n for n in tensor_names)
                has_old_attn = any("inner_attn_ln" in n for n in tensor_names)
                has_new_ffn = any("ffn_sub_norm" in n for n in tensor_names)
                has_old_ffn = any("ffn_layernorm" in n for n in tensor_names)

                has_attn_sub_norm = has_new_attn or has_old_attn
                has_ffn_sub_norm = has_new_ffn or has_old_ffn

                if has_old_attn:
                    attn_sub_norm_name = "self_attn.inner_attn_ln"
                if has_old_ffn:
                    ffn_sub_norm_name = "mlp.ffn_layernorm"

                # Build component order based on detected sub-norms
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

                arch_info = ArchInfo(
                    arch_type=arch_info.arch_type,
                    activation=arch_info.activation,
                    has_attn_sub_norm=has_attn_sub_norm,
                    has_ffn_sub_norm=has_ffn_sub_norm,
                    component_order=component_order,
                    embedding_pattern=arch_info.embedding_pattern,
                    final_norm_pattern=arch_info.final_norm_pattern,
                    layer_pattern=arch_info.layer_pattern,
                    has_qkv_bias=arch_info.has_qkv_bias,
                )
            except FileNotFoundError:
                pass  # Pre-quantized models have no safetensors; defaults are fine

        # Extract dimensions (BitNet defaults)
        hidden_dim = config.get("hidden_size", 2560)
        intermediate_dim = config.get("intermediate_size", 6912)
        num_layers = config.get("num_hidden_layers", 30)
        num_heads = config.get("num_attention_heads", 20)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        vocab_size = config.get("vocab_size", 128256)
        head_dim = config.get("head_dim", hidden_dim // num_heads)
        max_position_embeddings = config.get("max_position_embeddings", 4096)

        def align_to_128(dim):
            return ((dim + 127) // 128) * 128

        hidden_dim_orig = hidden_dim
        intermediate_dim_orig = intermediate_dim

        hidden_dim = align_to_128(hidden_dim)
        intermediate_dim = align_to_128(intermediate_dim)

        # Normalization epsilon
        norm_eps = config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-6))

        # RoPE theta
        rope_theta = config.get("rope_theta", 10000.0)

        # Activation function from config.json (overrides architecture default)
        hidden_act = config.get("hidden_act", None)
        if hidden_act:
            act_map = {
                "silu": ActivationType.SILU,
                "swish": ActivationType.SILU,
                "relu_squared": ActivationType.RELU_SQR,
                "relu2": ActivationType.RELU_SQR,
            }
            if hidden_act.lower() in act_map:
                detected_activation = act_map[hidden_act.lower()]
                if detected_activation != arch_info.activation:
                    arch_info = ArchInfo(
                        arch_type=arch_info.arch_type,
                        activation=detected_activation,
                        has_attn_sub_norm=arch_info.has_attn_sub_norm,
                        has_ffn_sub_norm=arch_info.has_ffn_sub_norm,
                        component_order=arch_info.component_order,
                        embedding_pattern=arch_info.embedding_pattern,
                        final_norm_pattern=arch_info.final_norm_pattern,
                        layer_pattern=arch_info.layer_pattern,
                        has_qkv_bias=arch_info.has_qkv_bias,
                    )
            else:
                supported = list(act_map.keys())
                raise ValueError(
                    f"Unknown activation function '{hidden_act}'. Supported: {supported}"
                )

        # QKV bias detection from config.json
        has_qkv_bias = config.get("attention_bias", arch_info.has_qkv_bias)

        # Override has_qkv_bias from actual tensors if model_dir provided
        if model_dir:
            try:
                all_names = _get_all_tensor_names(model_dir)
                if any(
                    "q_proj.bias" in n or "k_proj.bias" in n or "v_proj.bias" in n
                    for n in all_names
                ):
                    has_qkv_bias = True
            except FileNotFoundError:
                pass

        # Update arch_info with detected QKV bias
        if has_qkv_bias != arch_info.has_qkv_bias:
            arch_info = ArchInfo(
                arch_type=arch_info.arch_type,
                activation=arch_info.activation,
                has_attn_sub_norm=arch_info.has_attn_sub_norm
                if not model_dir
                else has_attn_sub_norm,
                has_ffn_sub_norm=arch_info.has_ffn_sub_norm
                if not model_dir
                else has_ffn_sub_norm,
                component_order=arch_info.component_order,
                embedding_pattern=arch_info.embedding_pattern,
                final_norm_pattern=arch_info.final_norm_pattern,
                layer_pattern=arch_info.layer_pattern,
                has_qkv_bias=has_qkv_bias,
            )

        # Tied embeddings detection
        tie_word_embeddings = config.get("tie_word_embeddings", False)
        if model_dir:
            try:
                all_names = _get_all_tensor_names(model_dir)
                if any("lm_head.weight" in n for n in all_names):
                    tie_word_embeddings = False
            except FileNotFoundError:
                pass

        # EOS tokens - collect all stop tokens from config.json and tokenizer.json
        eos_tokens = []

        # config.json eos_token_id (authoritative source, may be a list)
        eot_defaults = {
            ArchType.BITNET: 128009,
            ArchType.LLAMA: 128009,
        }
        eot_raw = config.get("eos_token_id", eot_defaults.get(arch_info.arch_type, 2))
        if isinstance(eot_raw, list):
            eos_tokens.extend(eot_raw)
        else:
            eos_tokens.append(eot_raw)

        # Also collect stop tokens from tokenizer.json
        if model_dir:
            tokenizer_json_path = os.path.join(model_dir, "tokenizer.json")
            if os.path.exists(tokenizer_json_path):
                try:
                    with open(tokenizer_json_path, encoding="utf-8") as f:
                        tokenizer_data = json.load(f)
                    stop_names = (
                        "<|eot_id|>",
                        "<|im_end|>",
                        "<|end_of_text|>",
                        "<|endoftext|>",
                        "</s>",
                    )
                    for token_entry in tokenizer_data.get("added_tokens", []):
                        if token_entry.get("content", "") in stop_names:
                            eos_tokens.append(token_entry["id"])
                except (json.JSONDecodeError, IOError, KeyError):
                    pass

        # Also collect stop tokens from LoRA adapter files if present
        if adapter_dir:
            # Check lora_tokenizer.json for stop tokens
            lora_tokenizer_path = os.path.join(adapter_dir, "lora_tokenizer.json")
            if os.path.exists(lora_tokenizer_path):
                try:
                    with open(lora_tokenizer_path, encoding="utf-8") as f:
                        lora_tok_data = json.load(f)
                    stop_names = (
                        "<|eot_id|>",
                        "<|im_end|>",
                        "<|end_of_text|>",
                        "<|endoftext|>",
                        "</s>",
                    )
                    for token_entry in lora_tok_data.get("added_tokens", []):
                        if token_entry.get("content", "") in stop_names:
                            eos_tokens.append(token_entry["id"])
                except (json.JSONDecodeError, IOError, KeyError):
                    pass

            # Check lora_tokenizer_config.json for eos_token override
            lora_tok_cfg_path = os.path.join(adapter_dir, "lora_tokenizer_config.json")
            if os.path.exists(lora_tok_cfg_path):
                try:
                    with open(lora_tok_cfg_path, encoding="utf-8") as f:
                        lora_tok_cfg = json.load(f)
                    eos_token_str = lora_tok_cfg.get("eos_token")
                    if eos_token_str:
                        # Resolve token string to ID using lora_tokenizer.json or base tokenizer.json
                        for tok_path in (
                            lora_tokenizer_path,
                            os.path.join(model_dir, "tokenizer.json"),
                        ):
                            if os.path.exists(tok_path):
                                with open(tok_path, encoding="utf-8") as f:
                                    tok_data = json.load(f)
                                for token_entry in tok_data.get("added_tokens", []):
                                    if token_entry.get("content") == eos_token_str:
                                        eos_tokens.append(token_entry["id"])
                                        break
                                else:
                                    continue
                                break
                except (json.JSONDecodeError, IOError, KeyError):
                    pass

        # Deduplicate while preserving order
        seen = set()
        eos_tokens = [t for t in eos_tokens if not (t in seen or seen.add(t))]

        return cls(
            arch_type=arch_info.arch_type,
            arch_info=arch_info,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            eos_tokens=eos_tokens,
            has_qkv_bias=has_qkv_bias,
            tie_word_embeddings=tie_word_embeddings,
            hidden_dim_orig=hidden_dim_orig,
            intermediate_dim_orig=intermediate_dim_orig,
        )
