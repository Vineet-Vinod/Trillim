# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for model architecture extraction helpers."""

import json
import struct
import tempfile
from pathlib import Path
import unittest

from trillim.model_arch import (
    ARCH_REGISTRY,
    ActivationType,
    ArchType,
    ModelConfig,
    _collect_eos_tokens,
    _get_all_tensor_names,
)


class ModelArchTests(unittest.TestCase):
    def _write_json(self, path: Path, data) -> Path:
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _write_config(self, root: Path, data: dict) -> Path:
        return self._write_json(root / "config.json", data)

    def _write_tensor_index(self, model_dir: Path, *tensor_names: str) -> None:
        self._write_json(
            model_dir / "model.safetensors.index.json",
            {"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensor_names}},
        )

    def test_from_config_json_rejects_unsupported_architecture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                {"architectures": ["MissingForCausalLM"]},
            )

            with self.assertRaisesRegex(ValueError, "Unsupported architecture 'MissingForCausalLM'"):
                ModelConfig.from_config_json(str(config_path))

    def test_from_config_json_uses_old_bitnet_sub_norms_and_tensor_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            config_path = self._write_config(
                model_dir,
                {
                    "architectures": ["BitnetForCausalLM"],
                    "hidden_size": 2570,
                    "intermediate_size": 7000,
                    "num_attention_heads": 10,
                    "num_hidden_layers": 12,
                    "vocab_size": 32000,
                    "tie_word_embeddings": True,
                },
            )
            self._write_tensor_index(
                model_dir,
                "model.layers.0.self_attn.inner_attn_ln.weight",
                "model.layers.0.mlp.ffn_layernorm.weight",
                "model.layers.0.self_attn.q_proj.bias",
                "lm_head.weight",
            )

            config = ModelConfig.from_config_json(str(config_path), model_dir=str(model_dir))

        self.assertEqual(config.arch_type, ArchType.BITNET)
        self.assertEqual(config.arch_info.activation, ActivationType.RELU_SQR)
        self.assertTrue(config.arch_info.has_attn_sub_norm)
        self.assertTrue(config.arch_info.has_ffn_sub_norm)
        self.assertEqual(
            config.arch_info.component_order,
            [
                "input_layernorm",
                "self_attn.inner_attn_ln",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_proj",
                "self_attn.o_proj",
                "post_attention_layernorm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.ffn_layernorm",
                "mlp.down_proj",
            ],
        )
        self.assertTrue(config.has_qkv_bias)
        self.assertTrue(config.arch_info.has_qkv_bias)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.hidden_dim, 2688)
        self.assertEqual(config.intermediate_dim, 7040)
        self.assertEqual(config.hidden_dim_orig, 2570)
        self.assertEqual(config.intermediate_dim_orig, 7000)
        self.assertEqual(config.head_dim, 257)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.vocab_size, 32000)

    def test_from_config_json_drops_bitnet_sub_norms_when_tensors_do_not_have_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            config_path = self._write_config(
                model_dir,
                {"architectures": ["BitnetForCausalLM"]},
            )
            self._write_tensor_index(model_dir)

            config = ModelConfig.from_config_json(str(config_path), model_dir=str(model_dir))

        self.assertFalse(config.arch_info.has_attn_sub_norm)
        self.assertFalse(config.arch_info.has_ffn_sub_norm)
        self.assertEqual(
            config.arch_info.component_order,
            [
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
        )

    def test_from_config_json_keeps_defaults_when_tensor_scan_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            config_path = self._write_config(
                model_dir,
                {"architectures": ["BitnetForCausalLM"], "tie_word_embeddings": True},
            )

            config = ModelConfig.from_config_json(str(config_path), model_dir=str(model_dir))

        self.assertTrue(config.arch_info.has_attn_sub_norm)
        self.assertTrue(config.arch_info.has_ffn_sub_norm)
        self.assertEqual(
            config.arch_info.component_order,
            ARCH_REGISTRY["bitnetforcausallm"].component_order,
        )
        self.assertTrue(config.tie_word_embeddings)

    def test_from_config_json_applies_activation_override_and_scalar_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                {
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_act": "relu2",
                    "hidden_size": 2570,
                    "intermediate_size": 7000,
                    "num_attention_heads": 10,
                    "num_key_value_heads": 4,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 500000.0,
                    "partial_rotary_factor": 0.5,
                    "max_position_embeddings": 8192,
                },
            )

            config = ModelConfig.from_config_json(str(config_path))

        self.assertEqual(config.arch_info.activation, ActivationType.RELU_SQR)
        self.assertEqual(config.hidden_dim, 2688)
        self.assertEqual(config.intermediate_dim, 7040)
        self.assertEqual(config.num_kv_heads, 4)
        self.assertEqual(config.max_position_embeddings, 8192)
        self.assertEqual(config.norm_eps, 1e-5)
        self.assertEqual(config.rope_theta, 500000.0)
        self.assertEqual(config.partial_rotary_factor, 0.5)

    def test_from_config_json_keeps_arch_default_activation_when_override_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                {
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_act": "swish",
                },
            )

            config = ModelConfig.from_config_json(str(config_path))

        self.assertEqual(config.arch_info.activation, ActivationType.SILU)

    def test_from_config_json_rejects_unknown_activation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self._write_config(
                Path(temp_dir),
                {
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_act": "gelu",
                },
            )

            with self.assertRaisesRegex(ValueError, "Unknown activation function 'gelu'"):
                ModelConfig.from_config_json(str(config_path))

    def test_from_config_json_supports_qwen35_text_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            config_path = self._write_config(
                model_dir,
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "text_config": {
                        "hidden_size": 2560,
                        "intermediate_size": 9216,
                        "num_hidden_layers": 32,
                        "num_attention_heads": 16,
                        "num_key_value_heads": 4,
                        "head_dim": 256,
                        "layer_types": ["linear_attention", "full_attention"],
                        "attn_output_gate": True,
                        "linear_num_key_heads": 16,
                        "linear_num_value_heads": 32,
                        "linear_key_head_dim": 128,
                        "linear_value_head_dim": 128,
                        "linear_conv_kernel_dim": 4,
                        "vocab_size": 248320,
                        "max_position_embeddings": 262144,
                        "rms_norm_eps": 1e-6,
                        "rope_parameters": {
                            "rope_theta": 10000000.0,
                            "partial_rotary_factor": 0.25,
                        },
                        "hidden_act": "silu",
                        "eos_token_id": 248044,
                        "tie_word_embeddings": True,
                        "attention_bias": False,
                    },
                },
            )
            self._write_tensor_index(
                model_dir,
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.norm.weight",
            )

            config = ModelConfig.from_config_json(str(config_path), model_dir=str(model_dir))

        self.assertEqual(config.arch_type, ArchType.QWEN35)
        self.assertEqual(config.arch_info.activation, ActivationType.SILU)
        self.assertEqual(config.arch_info.embedding_pattern, "language_model.embed_tokens")
        self.assertEqual(config.arch_info.final_norm_pattern, "model.language_model.norm.weight")
        self.assertEqual(config.hidden_dim, 2560)
        self.assertEqual(config.intermediate_dim, 9216)
        self.assertEqual(config.num_layers, 32)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.num_kv_heads, 4)
        self.assertEqual(config.vocab_size, 248320)
        self.assertEqual(config.head_dim, 256)
        self.assertEqual(config.max_position_embeddings, 262144)
        self.assertEqual(config.norm_eps, 1e-6)
        self.assertEqual(config.rope_theta, 10000000.0)
        self.assertEqual(config.partial_rotary_factor, 0.25)
        self.assertTrue(config.tie_word_embeddings)
        self.assertFalse(config.has_qkv_bias)
        self.assertEqual(config.eos_tokens, [248044])
        self.assertEqual(config.layer_types, ["linear_attention", "full_attention"])
        self.assertTrue(config.attn_output_gate)
        self.assertEqual(config.linear_num_key_heads, 16)
        self.assertEqual(config.linear_num_value_heads, 32)
        self.assertEqual(config.linear_key_head_dim, 128)
        self.assertEqual(config.linear_value_head_dim, 128)
        self.assertEqual(config.linear_conv_kernel_dim, 4)

    def test_from_config_json_collects_and_dedupes_eos_tokens_from_model_and_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            config_path = self._write_config(
                model_dir,
                {
                    "architectures": ["LlamaForCausalLM"],
                    "eos_token_id": [2, 10],
                },
            )
            self._write_json(
                model_dir / "tokenizer.json",
                {
                    "added_tokens": [
                        {"content": "<|eot_id|>", "id": 10},
                        {"content": "</s>", "id": 11},
                        {"content": "<base_eos>", "id": 20},
                    ]
                },
            )
            self._write_json(
                adapter_dir / "lora_tokenizer.json",
                {
                    "added_tokens": [
                        {"content": "<|im_end|>", "id": 12},
                        {"content": "</s>", "id": 11},
                        {"content": "<lora_eos>", "id": 14},
                    ]
                },
            )
            self._write_json(
                adapter_dir / "lora_tokenizer_config.json",
                {"eos_token": "<lora_eos>"},
            )

            config = ModelConfig.from_config_json(
                str(config_path),
                model_dir=str(model_dir),
                adapter_dir=str(adapter_dir),
            )

        self.assertEqual(config.eos_tokens, [2, 10, 11, 12, 14])

    def test_collect_eos_tokens_falls_back_to_base_tokenizer_for_adapter_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            self._write_json(
                model_dir / "tokenizer.json",
                {"added_tokens": [{"content": "<base_eos>", "id": 20}]},
            )
            self._write_json(
                adapter_dir / "lora_tokenizer_config.json",
                {"eos_token": "<base_eos>"},
            )

            eos_tokens = _collect_eos_tokens(
                {},
                ArchType.LLAMA,
                str(model_dir),
                str(adapter_dir),
            )

        self.assertEqual(eos_tokens, [128009, 20])

    def test_collect_eos_tokens_ignores_invalid_json_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            (model_dir / "tokenizer.json").write_text("{invalid", encoding="utf-8")
            (adapter_dir / "lora_tokenizer.json").write_text("{invalid", encoding="utf-8")
            (adapter_dir / "lora_tokenizer_config.json").write_text("{invalid", encoding="utf-8")

            eos_tokens = _collect_eos_tokens(
                {},
                ArchType.LLAMA,
                str(model_dir),
                str(adapter_dir),
            )

        self.assertEqual(eos_tokens, [128009])

    def test_collect_eos_tokens_skips_unmatched_adapter_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            self._write_json(
                model_dir / "tokenizer.json",
                {"added_tokens": [{"content": "<base_eos>", "id": 20}]},
            )
            self._write_json(
                adapter_dir / "lora_tokenizer_config.json",
                {"eos_token": "<missing>"},
            )

            eos_tokens = _collect_eos_tokens(
                {},
                ArchType.LLAMA,
                str(model_dir),
                str(adapter_dir),
            )

        self.assertEqual(eos_tokens, [128009])

    def test_get_all_tensor_names_reads_unsharded_safetensors_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            header = {
                "__metadata__": {"format": "pt"},
                "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16"},
                "model.layers.0.mlp.down_proj.weight": {"dtype": "F16"},
            }
            header_bytes = json.dumps(header).encode("utf-8")
            with open(model_dir / "model.safetensors", "wb") as f:
                f.write(struct.pack("<Q", len(header_bytes)))
                f.write(header_bytes)

            tensor_names = _get_all_tensor_names(str(model_dir))

        self.assertEqual(
            tensor_names,
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            ],
        )


if __name__ == "__main__":
    unittest.main()
