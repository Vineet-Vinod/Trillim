"""Tests for local quantization helpers and entrypoints."""

from __future__ import annotations

import io
import json
import math
import os
import struct
import tempfile
import tomllib
import types
import unittest
import warnings
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer

from trillim._bundle_metadata import (
    CURRENT_FORMAT_VERSION,
    compute_base_model_config_hash,
)
from trillim.quantize import quantize
from trillim.quantize import _config as quantize_config
from trillim.quantize import _entrypoint as entrypoint
from trillim.quantize import _manifest as manifest
from trillim.quantize import _output as output
from tests.components.llm.support import patched_model_store


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    for key, (dtype, shape) in tensors.items():
        data_size = _dtype_size(dtype) * math.prod(shape)
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + data_size],
        }
        offset += data_size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + (b"\0" * offset))


def _read_manifest(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        shard_count = struct.unpack("<H", handle.read(2))[0]
        shards = []
        for _ in range(shard_count):
            length = struct.unpack("<H", handle.read(2))[0]
            shards.append(handle.read(length).decode("utf-8"))

        tensor_count = struct.unpack("<I", handle.read(4))[0]
        tensors = []
        for _ in range(tensor_count):
            tensors.append(
                {
                    "action": struct.unpack("<B", handle.read(1))[0],
                    "dtype": struct.unpack("<B", handle.read(1))[0],
                    "row": struct.unpack("<I", handle.read(4))[0],
                    "col": struct.unpack("<I", handle.read(4))[0],
                    "padded_row": struct.unpack("<I", handle.read(4))[0],
                    "padded_col": struct.unpack("<I", handle.read(4))[0],
                    "shard_idx": struct.unpack("<H", handle.read(2))[0],
                    "data_offset": struct.unpack("<Q", handle.read(8))[0],
                    "data_size": struct.unpack("<Q", handle.read(8))[0],
                    "has_scale": struct.unpack("<B", handle.read(1))[0],
                    "scale_shard_idx": struct.unpack("<H", handle.read(2))[0],
                    "scale_offset": struct.unpack("<Q", handle.read(8))[0],
                    "scale_size": struct.unpack("<Q", handle.read(8))[0],
                }
            )

        section_count = struct.unpack("<I", handle.read(4))[0]
        sections = []
        for _ in range(section_count):
            sections.append(
                {
                    "type": struct.unpack("<B", handle.read(1))[0],
                    "first_tensor_idx": struct.unpack("<I", handle.read(4))[0],
                    "num_tensors": struct.unpack("<I", handle.read(4))[0],
                }
            )
        remainder = handle.read()
    if not remainder:
        return {"shards": shards, "tensors": tensors, "sections": sections, "lora": None}
    offset = 0

    def _read(fmt: str):
        nonlocal offset
        size = struct.calcsize(fmt)
        value = struct.unpack(fmt, remainder[offset : offset + size])[0]
        offset += size
        return value

    num_layers = _read("<I")
    targets_per_layer = _read("<I")
    scale = _read("<d")
    layers = []
    for _ in range(num_layers):
        layer_entries = []
        for _ in range(targets_per_layer):
            present = _read("<B")
            if not present:
                layer_entries.append(None)
                continue
            layer_entries.append(
                {
                    "a_dtype": _read("<B"),
                    "a_rows": _read("<I"),
                    "a_cols": _read("<I"),
                    "a_shard_idx": _read("<H"),
                    "a_offset": _read("<Q"),
                    "a_size": _read("<Q"),
                    "b_dtype": _read("<B"),
                    "b_rows": _read("<I"),
                    "b_cols": _read("<I"),
                    "b_shard_idx": _read("<H"),
                    "b_offset": _read("<Q"),
                    "b_size": _read("<Q"),
                }
            )
        layers.append(layer_entries)
    return {
        "shards": shards,
        "tensors": tensors,
        "sections": sections,
        "lora": {
            "num_layers": num_layers,
            "targets_per_layer": targets_per_layer,
            "scale": scale,
            "layers": layers,
        },
    }


def _write_llama_model(root: Path, *, sharded: bool = False) -> Path:
    model_dir = root / "llama-source"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "Org/BaseModel",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 250,
                "intermediate_size": 300,
                "num_attention_heads": 5,
                "num_hidden_layers": 1,
                "num_key_value_heads": 5,
                "vocab_size": 256,
                "max_position_embeddings": 2048,
                "rope_theta": 7000.0,
                "hidden_act": "silu",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "generation_config.json").write_text(
        json.dumps({"temperature": 0.3}),
        encoding="utf-8",
    )
    tensors = {
        "model.embed_tokens.weight": ("F16", (4, 250)),
        "model.layers.0.input_layernorm.weight": ("F16", (250,)),
        "model.layers.0.self_attn.q_proj.weight": ("F16", (250, 250)),
        "model.layers.0.self_attn.k_proj.weight": ("I8", (250, 250)),
        "model.layers.0.self_attn.k_proj.weight_scale": ("F32", (250,)),
        "model.layers.0.mlp.gate_proj.weight": ("F16", (300, 250)),
        "model.layers.0.mlp.down_proj.bias": ("F16", (250,)),
        "model.norm.weight": ("F16", (250,)),
        "lm_head.weight": ("F16", (4, 250)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": ("F32", (8,)),
    }
    if sharded:
        shard1 = model_dir / "model-00001-of-00002.safetensors"
        shard2 = model_dir / "model-00002-of-00002.safetensors"
        shard1_tensors = {
            key: value
            for key, value in tensors.items()
            if key
            in {
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.k_proj.weight_scale",
            }
        }
        shard2_tensors = {key: value for key, value in tensors.items() if key not in shard1_tensors}
        _write_safetensors(shard1, shard1_tensors)
        _write_safetensors(shard2, shard2_tensors)
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        **{key: shard1.name for key in shard1_tensors},
                        **{key: shard2.name for key in shard2_tensors},
                    }
                }
            ),
            encoding="utf-8",
        )
    else:
        _write_safetensors(model_dir / "model.safetensors", tensors)
    return model_dir


def _write_qwen_multimodal_model(root: Path) -> Path:
    model_dir = root / "qwen-mm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "hidden_size": 2560,
                    "intermediate_size": 9216,
                    "num_attention_heads": 16,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 4,
                    "head_dim": 160,
                    "vocab_size": 248320,
                    "max_position_embeddings": 262144,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.25,
                    },
                    "tie_word_embeddings": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_safetensors(
        model_dir / "model.safetensors",
        {
            "model.language_model.embed_tokens.weight": ("F16", (16, 2560)),
            "model.language_model.layers.0.input_layernorm.weight": ("F16", (2560,)),
            "model.language_model.layers.0.self_attn.q_proj.weight": ("F16", (2560, 2560)),
            "model.language_model.layers.0.self_attn.k_proj.weight": ("F16", (640, 2560)),
            "model.language_model.layers.0.self_attn.v_proj.weight": ("F16", (640, 2560)),
            "model.language_model.layers.0.self_attn.o_proj.weight": ("F16", (2560, 2560)),
            "model.language_model.layers.0.mlp.gate_proj.weight": ("F16", (9216, 2560)),
            "model.language_model.layers.0.mlp.up_proj.weight": ("F16", (9216, 2560)),
            "model.language_model.layers.0.mlp.down_proj.weight": ("F16", (2560, 9216)),
            "model.language_model.norm.weight": ("F16", (2560,)),
            "model.visual.patch_embed.proj.weight": ("F16", (1024, 3, 2, 2)),
            "mtp.fc.weight": ("F16", (2560, 2560)),
        },
    )
    return model_dir


def _write_bonsai_source_model(
    root: Path,
    *,
    name: str = "bonsai-source",
    readme_text: str,
) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": 64,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "hidden_act": "silu",
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "README.md").write_text(readme_text, encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_safetensors(
        model_dir / "model.safetensors",
        {
            "model.embed_tokens.weight": ("F16", (8, 128)),
            "model.layers.0.input_layernorm.weight": ("F16", (128,)),
            "model.layers.0.self_attn.k_proj.weight": ("F16", (64, 128)),
            "model.layers.0.self_attn.v_proj.weight": ("F16", (64, 128)),
            "model.layers.0.self_attn.q_proj.weight": ("F16", (128, 128)),
            "model.layers.0.self_attn.q_norm.weight": ("F16", (32,)),
            "model.layers.0.self_attn.k_norm.weight": ("F16", (32,)),
            "model.layers.0.self_attn.o_proj.weight": ("F16", (128, 128)),
            "model.layers.0.post_attention_layernorm.weight": ("F16", (128,)),
            "model.layers.0.mlp.gate_proj.weight": ("F16", (256, 128)),
            "model.layers.0.mlp.up_proj.weight": ("F16", (256, 128)),
            "model.layers.0.mlp.down_proj.weight": ("F16", (128, 256)),
            "model.norm.weight": ("F16", (128,)),
            "lm_head.weight": ("F16", (8, 128)),
        },
    )
    return model_dir


def _write_adapter(
    root: Path,
    *,
    target_modules: list[str] | None = None,
    tensors: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> Path:
    adapter_dir = root / "adapter-source"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 4,
                "lora_alpha": 8,
                "target_modules": target_modules or ["q_proj", "gate_proj"],
                "base_model_name_or_path": "Org/BaseModel",
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}),
        encoding="utf-8",
    )
    (adapter_dir / "chat_template.jinja").write_text("{{ adapter }}", encoding="utf-8")
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        tensors
        or {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": ("F32", (4, 250)),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": ("F32", (250, 4)),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": ("F32", (4, 250)),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": ("F32", (300, 4)),
        },
    )
    return adapter_dir


def _write_config_only_model(
    root: Path,
    *,
    name: str = "config-only",
    payload: dict[str, object] | None = None,
) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            payload
            or {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 130,
                "intermediate_size": 129,
                "num_attention_heads": 5,
                "num_hidden_layers": 2,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def _write_remote_code_model(
    root: Path,
    *,
    name: str,
    config_payload: dict[str, object] | None,
    tokenizer_payload: dict[str, object] | None = None,
    files: dict[str, str],
) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    if config_payload is not None:
        (model_dir / "config.json").write_text(
            json.dumps(config_payload),
            encoding="utf-8",
        )
    if tokenizer_payload is not None:
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_payload),
            encoding="utf-8",
        )
    for relative_path, content in files.items():
        path = model_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return model_dir


def _dtype_size(dtype: str) -> int:
    return {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1}[dtype]


def _expected_lora_dims_for_test(
    config: entrypoint.ModelQuantizeConfig,
    target: str,
) -> tuple[int, int]:
    attention_output_dim = config.num_heads * config.head_dim
    key_value_output_dim = config.num_kv_heads * config.head_dim
    if target == "self_attn.q_proj":
        return config.hidden_dim_orig, attention_output_dim
    if target in {"self_attn.k_proj", "self_attn.v_proj"}:
        return config.hidden_dim_orig, key_value_output_dim
    if target == "self_attn.o_proj":
        return attention_output_dim, config.hidden_dim_orig
    if target in {"mlp.gate_proj", "mlp.up_proj"}:
        return config.hidden_dim_orig, config.intermediate_dim_orig
    if target == "mlp.down_proj":
        return config.intermediate_dim_orig, config.hidden_dim_orig
    raise AssertionError(f"Unsupported target in test: {target}")


class QuantizeTests(unittest.TestCase):
    def test_compute_base_model_config_hash_rejects_invalid_head_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            base_config = {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 250,
                "intermediate_size": 300,
                "num_hidden_layers": 1,
                "num_key_value_heads": 5,
                "vocab_size": 256,
            }
            for invalid_value in (True, "bad", 0):
                with self.subTest(invalid_value=invalid_value):
                    config_path.write_text(
                        json.dumps(dict(base_config, num_attention_heads=invalid_value)),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "num_attention_heads must be a positive integer",
                    ):
                        compute_base_model_config_hash(root)
            config_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "Model config must be a JSON object",
            ):
                compute_base_model_config_hash(root)

    def test_load_model_config_rejects_non_object_config_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            (model_dir / "config.json").write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "Model config must be a JSON object",
            ):
                entrypoint.load_model_config(model_dir)

    def test_build_manifest_supports_sharded_models_padding_and_scales(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _write_llama_model(Path(temp_dir), sharded=True)
            config = entrypoint.load_model_config(model_dir)
            manifest_path = manifest.build_manifest(
                model_dir,
                config,
                output_dir=model_dir,
                language_model_only=False,
            )

            payload = _read_manifest(manifest_path)

        self.assertEqual(len(payload["shards"]), 2)
        self.assertEqual(
            payload["sections"],
            [{"type": manifest.SECTION_TEXT_CORE, "first_tensor_idx": 0, "num_tensors": 8}],
        )
        embedding_entry = payload["tensors"][0]
        repack_entry = next(entry for entry in payload["tensors"] if entry["has_scale"] == 1)
        padded_entry = next(
            entry for entry in payload["tensors"] if entry["row"] == 300 and entry["col"] == 250
        )
        self.assertEqual(embedding_entry["action"], manifest.ACTION_BF16_RAW)
        self.assertEqual(repack_entry["action"], manifest.ACTION_REPACK_TERNARY)
        self.assertEqual((padded_entry["padded_row"], padded_entry["padded_col"]), (384, 256))

    def test_bonsai_readme_discriminator_controls_manifest_and_bundle_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary_model_dir = _write_bonsai_source_model(
                root,
                name="bonsai-binary",
                readme_text="Bonsai 1-bit release model.",
            )
            ternary_model_dir = _write_bonsai_source_model(
                root,
                name="bonsai-ternary",
                readme_text="Ternary Bonsai release model.",
            )
            binary_config = entrypoint.load_model_config(binary_model_dir)
            ternary_config = entrypoint.load_model_config(ternary_model_dir)

            self.assertEqual(binary_config.arch_type, quantize_config.ArchitectureType.BONSAI)
            self.assertEqual(
                ternary_config.arch_type,
                quantize_config.ArchitectureType.BONSAI_TERNARY,
            )

            binary_manifest = _read_manifest(
                manifest.build_manifest(
                    binary_model_dir,
                    binary_config,
                    output_dir=binary_model_dir,
                    language_model_only=False,
                )
            )
            ternary_manifest = _read_manifest(
                manifest.build_manifest(
                    ternary_model_dir,
                    ternary_config,
                    output_dir=ternary_model_dir,
                    language_model_only=False,
                )
            )

            self.assertEqual(binary_manifest["tensors"][0]["action"], manifest.ACTION_Q1_0_128)
            self.assertEqual(
                ternary_manifest["tensors"][0]["action"],
                manifest.ACTION_GROUP_TERNARY_QUANTIZE,
            )
            self.assertEqual(binary_manifest["tensors"][-1]["action"], manifest.ACTION_Q1_0_128)
            self.assertEqual(
                ternary_manifest["tensors"][-1]["action"],
                manifest.ACTION_GROUP_TERNARY_QUANTIZE,
            )
            norm_entry = next(
                entry for entry in ternary_manifest["tensors"] if entry["row"] == 128 and entry["col"] == 1
            )
            self.assertEqual(norm_entry["action"], manifest.ACTION_BF16_RAW)

            binary_output_dir = root / "binary-out"
            ternary_output_dir = root / "ternary-out"
            binary_output_dir.mkdir()
            ternary_output_dir.mkdir()
            output.copy_model_support_files(binary_model_dir, binary_output_dir)
            output.copy_model_support_files(ternary_model_dir, ternary_output_dir)
            output.write_model_metadata(binary_output_dir, config=binary_config, model_dir=binary_model_dir)
            output.write_model_metadata(ternary_output_dir, config=ternary_config, model_dir=ternary_model_dir)

            binary_bundle_config = json.loads(
                (binary_output_dir / "config.json").read_text(encoding="utf-8")
            )
            ternary_bundle_config = json.loads(
                (ternary_output_dir / "config.json").read_text(encoding="utf-8")
            )
            binary_metadata = json.loads(
                (binary_output_dir / "trillim_config.json").read_text(encoding="utf-8")
            )
            ternary_metadata = json.loads(
                (ternary_output_dir / "trillim_config.json").read_text(encoding="utf-8")
            )

            self.assertEqual(binary_bundle_config["architectures"], ["Qwen3ForCausalLM"])
            self.assertEqual(
                ternary_bundle_config["architectures"],
                ["Qwen3ForCausalLM"],
            )
            self.assertEqual(binary_metadata["quantization"], "binary")
            self.assertEqual(ternary_metadata["quantization"], "grouped-ternary")

    def test_build_manifest_rejects_unsupported_tensor_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _write_llama_model(Path(temp_dir))
            _write_safetensors(
                model_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("F16", (4, 250)),
                    "model.layers.0.unknown_branch.weight": ("F16", (250, 250)),
                },
            )
            config = entrypoint.load_model_config(model_dir)
            with self.assertRaisesRegex(ValueError, "layer unsupported at this time"):
                manifest.build_manifest(
                    model_dir,
                    config,
                    output_dir=model_dir,
                    language_model_only=False,
                )

    def test_qwen_multimodal_uses_language_model_only_for_known_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = _write_qwen_multimodal_model(Path(temp_dir))
            config = entrypoint.load_model_config(model_dir)

            self.assertTrue(manifest.determine_language_model_only(model_dir, config))
            with self.assertRaisesRegex(ValueError, "text-only quantization"):
                manifest.build_manifest(
                    model_dir,
                    config,
                    output_dir=model_dir,
                    language_model_only=False,
                )

            payload = _read_manifest(
                manifest.build_manifest(
                    model_dir,
                    config,
                    output_dir=model_dir,
                    language_model_only=True,
                )
            )

        self.assertEqual(
            payload["sections"],
            [{"type": manifest.SECTION_TEXT_CORE, "first_tensor_idx": 0, "num_tensors": 10}],
        )
        self.assertEqual(len(payload["tensors"]), 10)

    def test_validate_adapter_source_and_manifest_reject_unsupported_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            config = entrypoint.load_model_config(model_dir)
            adapter_dir = _write_adapter(root)

            manifest.validate_adapter_source(adapter_dir, config)
            payload = _read_manifest(
                manifest.build_manifest(
                    model_dir,
                    config,
                    output_dir=adapter_dir,
                    adapter_dir=adapter_dir,
                    skip_model=True,
                    language_model_only=False,
                )
            )
            self.assertEqual(payload["lora"]["num_layers"], 1)
            self.assertEqual(payload["lora"]["targets_per_layer"], 7)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            config = entrypoint.load_model_config(model_dir)
            bad_adapter_dir = _write_adapter(root, target_modules=["q_proj", "embed_tokens"])
            with self.assertRaisesRegex(ValueError, "layer unsupported at this time"):
                manifest.validate_adapter_source(bad_adapter_dir, config)

    def test_validate_adapter_source_rejects_shape_mismatches_for_all_supported_targets(self):
        short_to_full = {
            "k_proj": "self_attn.k_proj",
            "v_proj": "self_attn.v_proj",
            "q_proj": "self_attn.q_proj",
            "o_proj": "self_attn.o_proj",
            "gate_proj": "mlp.gate_proj",
            "up_proj": "mlp.up_proj",
            "down_proj": "mlp.down_proj",
        }
        for short_target, full_target in short_to_full.items():
            with self.subTest(target=short_target, mismatch="lora_A"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    model_dir = _write_llama_model(root)
                    config_path = model_dir / "config.json"
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    payload["num_hidden_layers"] = 2
                    config_path.write_text(json.dumps(payload), encoding="utf-8")
                    config = entrypoint.load_model_config(model_dir)
                    expected_input_dim, expected_output_dim = _expected_lora_dims_for_test(
                        config,
                        full_target,
                    )
                    adapter_dir = _write_adapter(
                        root,
                        target_modules=[short_target],
                        tensors={
                            f"base_model.model.model.layers.1.{full_target}.lora_A.weight": (
                                "F32",
                                (4, expected_input_dim + 1),
                            ),
                            f"base_model.model.model.layers.1.{full_target}.lora_B.weight": (
                                "F32",
                                (expected_output_dim, 4),
                            ),
                        },
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        f"Adapter {full_target} lora_A on layer 1 has input dim",
                    ):
                        manifest.validate_adapter_source(adapter_dir, config)

            with self.subTest(target=short_target, mismatch="lora_B"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    model_dir = _write_llama_model(root)
                    config_path = model_dir / "config.json"
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    payload["num_hidden_layers"] = 2
                    if short_target in {"k_proj", "v_proj"}:
                        payload["num_key_value_heads"] = 2
                    config_path.write_text(json.dumps(payload), encoding="utf-8")
                    config = entrypoint.load_model_config(model_dir)
                    expected_input_dim, expected_output_dim = _expected_lora_dims_for_test(
                        config,
                        full_target,
                    )
                    adapter_dir = _write_adapter(
                        root,
                        target_modules=[short_target],
                        tensors={
                            f"base_model.model.model.layers.1.{full_target}.lora_A.weight": (
                                "F32",
                                (4, expected_input_dim),
                            ),
                            f"base_model.model.model.layers.1.{full_target}.lora_B.weight": (
                                "F32",
                                (expected_output_dim + 1, 4),
                            ),
                        },
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        f"Adapter {full_target} lora_B on layer 1 has output dim",
                    ):
                        manifest.validate_adapter_source(adapter_dir, config)

    def test_validate_adapter_source_rejects_rank_mismatches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            config = entrypoint.load_model_config(model_dir)
            adapter_dir = _write_adapter(
                root,
                target_modules=["q_proj"],
                tensors={
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
                        "F32",
                        (3, 250),
                    ),
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (
                        "F32",
                        (250, 4),
                    ),
                },
            )
            with self.assertRaisesRegex(ValueError, "declares r=4"):
                manifest.validate_adapter_source(adapter_dir, config)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            config = entrypoint.load_model_config(model_dir)
            adapter_dir = _write_adapter(
                root,
                target_modules=["q_proj"],
                tensors={
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
                        "F32",
                        (4, 250),
                    ),
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (
                        "F32",
                        (250, 3),
                    ),
                },
            )
            with self.assertRaisesRegex(ValueError, "declares r=4"):
                manifest.validate_adapter_source(adapter_dir, config)

    def test_expected_lora_dims_rejects_unsupported_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            config = entrypoint.load_model_config(model_dir)

            with self.assertRaisesRegex(ValueError, "layer unsupported at this time"):
                manifest._expected_lora_dims(config, "embed_tokens")

    def test_load_model_config_trusts_tied_embeddings_over_materialized_lm_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            config_path = model_dir / "config.json"
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["tie_word_embeddings"] = True
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            config = entrypoint.load_model_config(model_dir)
            tensors = manifest._ordered_text_tensors(
                manifest.get_tensor_metadata(model_dir / "model.safetensors"),
                config,
                language_model_only=False,
            )

        self.assertTrue(config.tie_word_embeddings)
        self.assertNotIn(
            "lm_head.weight",
            [str(item["key"]) for item in tensors],
        )

    def test_copy_policies_and_metadata_writers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            (model_dir / "weights.safetensors").write_bytes(b"raw")
            (model_dir / "unused.bin").write_bytes(b"skip")
            (model_dir / "tokenization_trillim.py").write_text(
                "from .shared import value\nclass Tokenizer: pass\n",
                encoding="utf-8",
            )
            (model_dir / "configuration_trillim.py").write_text(
                "from .shared import value\nclass Config: pass\n",
                encoding="utf-8",
            )
            (model_dir / "shared.py").write_text("value = 1\n", encoding="utf-8")
            config_payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            config_payload["auto_map"] = {"AutoTokenizer": "tokenization_trillim.Tokenizer"}
            config_payload["text_config"] = {"auto_map": {"AutoConfig": "configuration_trillim.Config"}}
            (model_dir / "config.json").write_text(
                json.dumps(config_payload),
                encoding="utf-8",
            )
            adapter_dir = _write_adapter(root)
            (adapter_dir / "notes.txt").write_text("keep", encoding="utf-8")
            (adapter_dir / "qmodel.lora").write_bytes(b"skip")

            model_output_dir = root / "model-out"
            adapter_output_dir = root / "adapter-out"
            model_output_dir.mkdir()
            adapter_output_dir.mkdir()
            config = entrypoint.load_model_config(model_dir)

            output.copy_model_support_files(model_dir, model_output_dir)
            output.copy_adapter_support_files(adapter_dir, adapter_output_dir)
            output.write_model_metadata(model_output_dir, config=config, model_dir=model_dir)
            output.write_adapter_metadata(
                adapter_output_dir,
                config=config,
                adapter_dir=adapter_dir,
                model_dir=model_dir,
            )

            self.assertTrue((model_output_dir / "config.json").is_file())
            self.assertTrue((model_output_dir / "tokenization_trillim.py").is_file())
            self.assertTrue((model_output_dir / "shared.py").is_file())
            self.assertFalse((model_output_dir / "configuration_trillim.py").exists())
            self.assertFalse((model_output_dir / "weights.safetensors").exists())
            self.assertFalse((model_output_dir / "unused.bin").exists())
            self.assertTrue((adapter_output_dir / "adapter_config.json").is_file())
            self.assertTrue((adapter_output_dir / "notes.txt").is_file())
            self.assertFalse((adapter_output_dir / "adapter_model.safetensors").exists())
            self.assertFalse((adapter_output_dir / "qmodel.lora").exists())

            model_metadata = json.loads((model_output_dir / "trillim_config.json").read_text(encoding="utf-8"))
            adapter_metadata = json.loads((adapter_output_dir / "trillim_config.json").read_text(encoding="utf-8"))

            self.assertEqual(model_metadata["format_version"], CURRENT_FORMAT_VERSION)
            self.assertEqual(adapter_metadata["format_version"], CURRENT_FORMAT_VERSION)
            self.assertEqual(model_metadata["type"], "model")
            self.assertEqual(adapter_metadata["type"], "lora_adapter")
            self.assertEqual(model_metadata["remote_code"], True)
            self.assertEqual(adapter_metadata["remote_code"], False)

    def test_copy_model_support_files_infers_local_tokenizer_auto_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_config_only_model(root, name="custom-tokenizer-model")
            (model_dir / "vocab.txt").write_text("<unk>\nhello\nworld\n", encoding="utf-8")
            (model_dir / "special_tokens_map.json").write_text(
                json.dumps({"unk_token": "<unk>"}),
                encoding="utf-8",
            )
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "ExampleTokenizer", "unk_token": "<unk>"}),
                encoding="utf-8",
            )
            (model_dir / "tokenization_example.py").write_text(
                "\n".join(
                    [
                        "from transformers.tokenization_utils import PreTrainedTokenizer",
                        "",
                        'VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}',
                        "",
                        "class ExampleTokenizer(PreTrainedTokenizer):",
                        "    vocab_files_names = VOCAB_FILES_NAMES",
                        '    model_input_names = ["input_ids", "attention_mask"]',
                        "",
                        '    def __init__(self, vocab_file, unk_token="<unk>", **kwargs):',
                        "        self.vocab_file = vocab_file",
                        "        with open(vocab_file, encoding='utf-8') as handle:",
                        "            tokens = [line.strip() for line in handle if line.strip()]",
                        "        self._token_to_id = {token: index for index, token in enumerate(tokens)}",
                        "        self._id_to_token = {index: token for token, index in self._token_to_id.items()}",
                        "        super().__init__(unk_token=unk_token, **kwargs)",
                        "",
                        "    @property",
                        "    def vocab_size(self):",
                        "        return len(self._token_to_id)",
                        "",
                        "    def get_vocab(self):",
                        "        return dict(self._token_to_id)",
                        "",
                        "    def _tokenize(self, text):",
                        "        return text.split()",
                        "",
                        "    def _convert_token_to_id(self, token):",
                        "        return self._token_to_id.get(token, self._token_to_id[self.unk_token])",
                        "",
                        "    def _convert_id_to_token(self, index):",
                        "        return self._id_to_token[index]",
                        "",
                        "    def save_vocabulary(self, save_directory, filename_prefix=None):",
                        "        return (self.vocab_file,)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            model_output_dir = root / "model-out"
            model_output_dir.mkdir()
            config = entrypoint.load_model_config(model_dir)

            output.copy_model_support_files(model_dir, model_output_dir)
            output.write_model_metadata(model_output_dir, config=config, model_dir=model_dir)

            copied_tokenizer_config = json.loads(
                (model_output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                copied_tokenizer_config["auto_map"]["AutoTokenizer"],
                ["tokenization_example.ExampleTokenizer", None],
            )
            self.assertTrue((model_output_dir / "tokenization_example.py").is_file())
            model_metadata = json.loads(
                (model_output_dir / "trillim_config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(model_metadata["remote_code"])

            tokenizer = AutoTokenizer.from_pretrained(
                model_output_dir,
                trust_remote_code=True,
                use_fast=False,
            )
            self.assertEqual(type(tokenizer).__name__, "ExampleTokenizer")

    def test_copy_model_support_files_ignores_missing_config_only_remote_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_config_only_model(
                root,
                name="stale-remote-model",
                payload={
                    "architectures": ["LlamaForCausalLM"],
                    "auto_map": {"AutoConfig": "configuration_missing.MissingConfig"},
                    "hidden_size": 130,
                    "intermediate_size": 129,
                    "num_attention_heads": 5,
                    "num_hidden_layers": 2,
                    "vocab_size": 32,
                },
            )
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
                encoding="utf-8",
            )
            model_output_dir = root / "model-out"
            model_output_dir.mkdir()
            config = entrypoint.load_model_config(model_dir)

            output.copy_model_support_files(model_dir, model_output_dir)
            output.write_model_metadata(model_output_dir, config=config, model_dir=model_dir)

            self.assertFalse((model_output_dir / "configuration_missing.py").exists())
            model_metadata = json.loads(
                (model_output_dir / "trillim_config.json").read_text(encoding="utf-8")
            )
            self.assertFalse(model_metadata["remote_code"])

    def test_prepare_output_target_prompts_dedups_and_recovers(self):
        with patched_model_store() as root:
            source_dir = root / "source"
            source_dir.mkdir()
            preferred = output.prepare_output_target(source_dir)
            self.assertEqual(preferred, root / "Local" / "source-TRNQ")

        with patched_model_store() as root, patch(
            "trillim.quantize._output.sys_stdin_isatty",
            return_value=False,
        ), patch(
            "trillim.quantize._output.sys_stdout_isatty",
            return_value=False,
        ):
            source_dir = root / "source"
            source_dir.mkdir()
            target = root / "Local" / "source-TRNQ"
            target.mkdir(parents=True)
            deduped = output.prepare_output_target(source_dir)
            self.assertEqual(deduped, root / "Local" / "source-TRNQ-2")

        with patched_model_store() as root, patch(
            "trillim.quantize._output.sys_stdin_isatty",
            return_value=True,
        ), patch(
            "trillim.quantize._output.sys_stdout_isatty",
            return_value=True,
        ), patch("builtins.input", side_effect=["maybe", "y"]):
            source_dir = root / "source"
            source_dir.mkdir()
            target = root / "Local" / "source-TRNQ"
            target.mkdir(parents=True)
            chosen = output.prepare_output_target(source_dir)
            self.assertEqual(chosen, target)

        with patched_model_store() as root:
            target = root / "Local" / "source-TRNQ"
            staging = root / "Local" / "source-TRNQ-new"
            backup = root / "Local" / "source-TRNQ-old"
            staging.mkdir(parents=True)
            backup.mkdir(parents=True)
            output.mark_staging_complete(staging)
            (staging / "qmodel.tensors").write_bytes(b"new")
            (backup / "qmodel.tensors").write_bytes(b"old")
            output.recover_publish_state(target)
            self.assertTrue((target / "qmodel.tensors").is_file())
            self.assertFalse(staging.exists())
            self.assertFalse(backup.exists())

        with patched_model_store() as root:
            target = root / "Local" / "source-TRNQ"
            staging = root / "Local" / "source-TRNQ-new"
            backup = root / "Local" / "source-TRNQ-old"
            staging.mkdir(parents=True)
            backup.mkdir(parents=True)
            (backup / "qmodel.tensors").write_bytes(b"old")
            output.recover_publish_state(target)
            self.assertTrue((target / "qmodel.tensors").is_file())
            self.assertFalse(staging.exists())
            self.assertFalse(backup.exists())

    def test_publish_staging_dir_rolls_back_old_bundle_on_rename_error(self):
        with patched_model_store() as root:
            target = root / "Local" / "bundle-TRNQ"
            target.mkdir(parents=True)
            (target / "qmodel.tensors").write_bytes(b"old")
            staging = root / "Local" / "bundle-TRNQ-new"
            staging.mkdir()
            output.mark_staging_complete(staging)
            rename_calls: list[tuple[Path, Path]] = []
            real_replace = output.os.replace

            def fake_replace(src, dst):
                rename_calls.append((Path(src), Path(dst)))
                if Path(src) == staging:
                    raise OSError("boom")
                return real_replace(src, dst)

            with patch("trillim.quantize._output.os.replace", side_effect=fake_replace):
                with self.assertRaises(OSError):
                    output.publish_staging_dir(target)

            self.assertTrue((target / "qmodel.tensors").is_file())
            self.assertFalse((root / "Local" / "bundle-TRNQ-old").exists())
            self.assertEqual(len(rename_calls), 3)

    def test_quantize_model_flow_publishes_local_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir, patched_model_store() as root:
            model_dir = _write_llama_model(Path(temp_dir))

            def fake_run_model_quantizer(_binary, _model_dir, _config, *, output_dir, language_model_only):
                self.assertFalse(language_model_only)
                (output_dir / "qmodel.tensors").write_bytes(b"model")
                (output_dir / "rope.cache").write_bytes(b"rope")

            with patch(
                "trillim.quantize._entrypoint.resolve_quantize_binary",
                return_value=Path("/tmp/binary"),
            ), patch(
                "trillim.quantize._entrypoint.run_model_quantizer",
                side_effect=fake_run_model_quantizer,
            ):
                result = quantize(model_dir)

            target = root / "Local" / "llama-source-TRNQ"
            self.assertEqual(result.bundle_path, target)
            self.assertEqual(result.bundle_type, "model")
            self.assertTrue((target / "qmodel.tensors").is_file())
            self.assertTrue((target / "rope.cache").is_file())
            self.assertEqual(
                json.loads((target / "trillim_config.json").read_text(encoding="utf-8"))["format_version"],
                CURRENT_FORMAT_VERSION,
            )

    def test_quantize_adapter_flow_warns_for_partial_multimodal_and_rejects_managed_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir, patched_model_store() as root:
            root_path = Path(temp_dir)
            model_dir = _write_qwen_multimodal_model(root_path)
            adapter_dir = _write_adapter(root_path)
            output_text = io.StringIO()

            def fake_run_adapter_quantizer(_binary, _model_dir, _config, *, adapter_dir, output_dir, language_model_only):
                self.assertEqual(adapter_dir.name, "adapter-source")
                self.assertTrue(language_model_only)
                (output_dir / "qmodel.lora").write_bytes(b"adapter")

            with patch(
                "trillim.quantize._entrypoint.resolve_quantize_binary",
                return_value=Path("/tmp/binary"),
            ), patch(
                "trillim.quantize._entrypoint.run_adapter_quantizer",
                side_effect=fake_run_adapter_quantizer,
            ), patch(
                "trillim.quantize._entrypoint.validate_adapter_source",
            ), redirect_stdout(output_text):
                result = quantize(model_dir, adapter_dir)

            target = root / "Local" / "adapter-source-TRNQ"
            self.assertEqual(result.bundle_path, target)
            self.assertTrue(result.used_language_model_only)
            self.assertTrue((target / "qmodel.lora").is_file())
            self.assertIn("text inference", output_text.getvalue())

        with patched_model_store() as root:
            managed_model = root / "Local" / "managed-model"
            managed_model.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                quantize(managed_model)

    def test_quantize_allows_grouped_ternary_bonsai_adapters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_bonsai_source_model(
                root,
                name="bonsai-ternary",
                readme_text="Ternary Bonsai release model.",
            )
            adapter_dir = _write_adapter(
                root,
                tensors={
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": ("F32", (4, 128)),
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": ("F32", (128, 4)),
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": ("F32", (4, 128)),
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": ("F32", (256, 4)),
                },
            )

            def fake_run_adapter_quantizer(
                _binary, _model_dir, _config, *, adapter_dir, output_dir, language_model_only
            ):
                del _binary, _model_dir, _config, adapter_dir, language_model_only
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "qmodel.lora").write_bytes(b"adapter")

            with patch(
                "trillim.quantize._entrypoint.resolve_quantize_binary",
                return_value=Path("/tmp/binary"),
            ), patch(
                "trillim.quantize._entrypoint.run_adapter_quantizer",
                side_effect=fake_run_adapter_quantizer,
            ), patch(
                "trillim.quantize._entrypoint.prepare_output_target",
                return_value=Path("/tmp/adapter-target"),
            ), patch(
                "trillim.quantize._entrypoint.build_staging_dir",
                return_value=Path("/tmp/adapter-staging"),
            ), patch(
                "trillim.quantize._entrypoint.copy_adapter_support_files",
            ), patch(
                "trillim.quantize._entrypoint.write_adapter_metadata",
            ), patch(
                "trillim.quantize._entrypoint.mark_staging_complete",
            ), patch(
                "trillim.quantize._entrypoint.publish_staging_dir",
            ):
                result = quantize(model_dir, adapter_dir)

            self.assertEqual(result.bundle_type, "adapter")


class QuantizeInternalTests(unittest.TestCase):
    def test_quantize_config_helpers_cover_defaults_aliases_and_parsing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_config_only_model(root)

            config = entrypoint.load_model_config(model_dir)

            self.assertEqual(config.hidden_dim, 256)
            self.assertEqual(config.intermediate_dim, 256)
            self.assertEqual(config.hidden_dim_orig, 130)
            self.assertEqual(config.intermediate_dim_orig, 129)
            self.assertEqual(config.num_kv_heads, 5)
            self.assertEqual(config.head_dim, 26)
            self.assertEqual(config.max_position_embeddings, 4096)
            self.assertEqual(config.rope_theta, 10000.0)
            self.assertEqual(config.partial_rotary_factor, 1.0)
            self.assertFalse(config.tie_word_embeddings)
            self.assertEqual(config.source_model, "")

            with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
                quantize_config._resolve_arch_info({"architectures": ["UnknownForCausalLM"]})

            self.assertEqual(quantize_config._align_to_128(129), 256)
            self.assertEqual(quantize_config._resolve_activation({}), "silu")
            self.assertEqual(
                quantize_config._resolve_activation({"hidden_act": "ReLU2"}),
                "relu_squared",
            )
            self.assertEqual(
                quantize_config._resolve_rope_theta({"rope_parameters": {"rope_theta": 777.0}}),
                777.0,
            )
            self.assertEqual(
                quantize_config._resolve_partial_rotary_factor(
                    {"rope_parameters": {"partial_rotary_factor": 0.25}}
                ),
                0.25,
            )
            self.assertEqual(
                quantize_config._resolve_partial_rotary_factor({"partial_rotary_factor": 0.5}),
                0.5,
            )
            self.assertFalse(quantize_config._resolve_tied_embeddings({}, None))
            self.assertEqual(
                quantize_config.layer_index_for_key(
                    "model.layers.12.mlp.gate_proj.weight",
                    config.arch_info,
                ),
                12,
            )
            self.assertIsNone(
                quantize_config.layer_index_for_key(
                    "model.embed_tokens.weight",
                    config.arch_info,
                )
            )

            with self.assertRaisesRegex(ValueError, "Unsupported activation function"):
                quantize_config._resolve_activation({"hidden_act": "gelu_fast"})
            for invalid_value in (True, "nope", 0):
                with self.subTest(invalid_value=invalid_value):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        quantize_config._require_positive_int(invalid_value, "bad")

    def test_quantize_config_load_tensor_names_and_legacy_bitnet_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_dir = root / "index-model"
            index_dir.mkdir()
            (index_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}),
                encoding="utf-8",
            )
            self.assertEqual(
                quantize_config._load_tensor_names_if_available(index_dir),
                ["a", "b"],
            )

            single_dir = root / "single-model"
            single_dir.mkdir()
            header = {
                "__metadata__": {"format": "pt"},
                "model.embed_tokens.weight": {
                    "dtype": "F16",
                    "shape": [2, 2],
                    "data_offsets": [0, 8],
                },
                "lm_head.weight": {
                    "dtype": "F16",
                    "shape": [2, 2],
                    "data_offsets": [8, 16],
                },
            }
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            (single_dir / "model.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + (b"\0" * 16)
            )
            self.assertEqual(
                quantize_config._load_tensor_names_if_available(single_dir),
                ["model.embed_tokens.weight", "lm_head.weight"],
            )

            invalid_index_dir = root / "invalid-index"
            invalid_index_dir.mkdir()
            (invalid_index_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": []}),
                encoding="utf-8",
            )
            self.assertIsNone(quantize_config._load_tensor_names_if_available(invalid_index_dir))
            self.assertIsNone(quantize_config._load_tensor_names_if_available(root / "missing"))

            bitnet_payload = {
                "architectures": ["BitnetForCausalLM"],
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "num_key_value_heads": 4,
                "vocab_size": 64,
            }
            bitnet_dir = _write_config_only_model(root, name="bitnet", payload=bitnet_payload)
            _write_safetensors(
                bitnet_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("F16", (4, 128)),
                    "model.layers.0.self_attn.inner_attn_ln.weight": ("F16", (128,)),
                    "model.layers.0.mlp.ffn_layernorm.weight": ("F16", (128,)),
                    "model.norm.weight": ("F16", (128,)),
                    "lm_head.weight": ("F16", (4, 128)),
                },
            )

            config = entrypoint.load_model_config(bitnet_dir)

            self.assertEqual(config.arch_name, "bitnet")
            self.assertEqual(config.arch_info.component_order[1], "self_attn.inner_attn_ln")
            self.assertEqual(config.arch_info.component_order[-2], "mlp.ffn_layernorm")

            clean_bitnet_root = root / "bitnet-clean-root"
            clean_bitnet_root.mkdir()
            clean_bitnet_dir = _write_config_only_model(
                clean_bitnet_root,
                payload=bitnet_payload,
            )
            _write_safetensors(
                clean_bitnet_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("F16", (4, 128)),
                    "model.layers.0.self_attn.attn_sub_norm.weight": ("F16", (128,)),
                    "model.layers.0.mlp.ffn_sub_norm.weight": ("F16", (128,)),
                    "model.norm.weight": ("F16", (128,)),
                    "lm_head.weight": ("F16", (4, 128)),
                },
            )
            clean_bitnet_config = entrypoint.load_model_config(clean_bitnet_dir)
            self.assertEqual(clean_bitnet_config.arch_info.component_order[1], "self_attn.attn_sub_norm")
            self.assertEqual(clean_bitnet_config.arch_info.component_order[-2], "mlp.ffn_sub_norm")

            old_attn_root = root / "bitnet-old-attn-root"
            old_attn_root.mkdir()
            old_attn_dir = _write_config_only_model(old_attn_root, payload=bitnet_payload)
            _write_safetensors(
                old_attn_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("F16", (4, 128)),
                    "model.layers.0.self_attn.inner_attn_ln.weight": ("F16", (128,)),
                    "model.layers.0.mlp.ffn_sub_norm.weight": ("F16", (128,)),
                    "model.norm.weight": ("F16", (128,)),
                    "lm_head.weight": ("F16", (4, 128)),
                },
            )
            old_attn_config = entrypoint.load_model_config(old_attn_dir)
            self.assertEqual(old_attn_config.arch_info.component_order[1], "self_attn.inner_attn_ln")
            self.assertEqual(old_attn_config.arch_info.component_order[-2], "mlp.ffn_sub_norm")

            old_ffn_root = root / "bitnet-old-ffn-root"
            old_ffn_root.mkdir()
            old_ffn_dir = _write_config_only_model(old_ffn_root, payload=bitnet_payload)
            _write_safetensors(
                old_ffn_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("F16", (4, 128)),
                    "model.layers.0.self_attn.attn_sub_norm.weight": ("F16", (128,)),
                    "model.layers.0.mlp.ffn_layernorm.weight": ("F16", (128,)),
                    "model.norm.weight": ("F16", (128,)),
                    "lm_head.weight": ("F16", (4, 128)),
                },
            )
            old_ffn_config = entrypoint.load_model_config(old_ffn_dir)
            self.assertEqual(old_ffn_config.arch_info.component_order[1], "self_attn.attn_sub_norm")
            self.assertEqual(old_ffn_config.arch_info.component_order[-2], "mlp.ffn_layernorm")

            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                entrypoint.load_model_config(root / "missing-config")

    def test_bonsai_readme_detection_warns_when_readme_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_bonsai_source_model(
                root,
                name="bonsai-missing-readme",
                readme_text="Bonsai 1-bit release model.",
            )
            (model_dir / "README.md").unlink()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = entrypoint.load_model_config(model_dir)

            self.assertEqual(config.arch_type, quantize_config.ArchitectureType.BONSAI)
            self.assertEqual(len(caught), 1)
            self.assertIn("defaulting Qwen3ForCausalLM Bonsai detection to binary", str(caught[0].message))

    def test_manifest_discovery_and_tensor_sorting_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root, sharded=True)
            config = entrypoint.load_model_config(model_dir)

            shard_files, weight_map = manifest.get_sharded_files(model_dir)
            self.assertEqual(len(shard_files), 2)
            self.assertIn("model.layers.0.self_attn.q_proj.weight", weight_map)
            self.assertEqual(
                manifest.get_all_tensor_names(model_dir),
                list(weight_map.keys()),
            )
            self.assertFalse(manifest.determine_language_model_only(model_dir, config))

            meta = manifest.get_tensor_metadata(shard_files[0])
            self.assertTrue(all("start" in item for item in meta))
            header, data_start = manifest._get_header_and_offsets(shard_files[0])
            self.assertIn("model.embed_tokens.weight", header)
            self.assertGreater(data_start, 8)
            self.assertEqual(manifest._safetensors_dtype_code("F16"), (manifest.DTYPE_F16, 2))
            with self.assertRaisesRegex(ValueError, "Unknown safetensors dtype"):
                manifest._safetensors_dtype_code("X9")

            metadata_path = root / "metadata-only.safetensors"
            metadata_header = {
                "__metadata__": {"format": "pt"},
                "visible.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
            }
            metadata_bytes = json.dumps(metadata_header, separators=(",", ":")).encode("utf-8")
            metadata_path.write_bytes(
                struct.pack("<Q", len(metadata_bytes)) + metadata_bytes + (b"\0" * 8)
            )
            self.assertEqual(
                manifest.get_tensor_metadata(metadata_path),
                [{"key": "visible.weight", "start": 0, "shape": [2, 2]}],
            )

            filtered = manifest._ordered_text_tensors(
                [
                    {"key": "model.layers.0.mlp.down_proj.bias", "shape": [250], "file": shard_files[1]},
                    {"key": "model.embed_tokens.weight", "shape": [4, 250], "file": shard_files[0]},
                    {"key": "lm_head.weight", "shape": [4, 250], "file": shard_files[1]},
                    {"key": "model.norm.weight", "shape": [250], "file": shard_files[1]},
                    {"key": "model.layers.0.self_attn.q_proj.weight", "shape": [250, 250], "file": shard_files[0]},
                    {"key": "model.layers.0.self_attn.rotary_emb.inv_freq", "shape": [8], "file": shard_files[1]},
                    {"key": "model.layers.0.self_attn.k_proj.weight_scale", "shape": [250], "file": shard_files[0]},
                    {"key": "model.visual.patch_embed.proj.weight", "shape": [1], "file": shard_files[1]},
                ],
                config,
                language_model_only=True,
            )
            self.assertEqual(
                [str(item["key"]) for item in filtered],
                [
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    "model.norm.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.mlp.down_proj.bias",
                ],
            )
            self.assertTrue(
                manifest._is_supported_text_tensor(
                    "model.layers.0.self_attn.q_proj.weight",
                    config,
                )
            )
            self.assertTrue(
                manifest._is_supported_text_tensor("lm_head.bias", config)
            )
            self.assertFalse(
                manifest._is_supported_text_tensor("model.layers.0.unknown.weight", config)
            )
            self.assertTrue(
                manifest._matches_component_key(
                    "model.layers.0.self_attn.q_proj.weight",
                    "self_attn.q_proj",
                )
            )
            self.assertTrue(
                manifest._matches_component_key(
                    "model.layers.0.input_layernorm.weight",
                    "input_layernorm",
                )
            )
            self.assertTrue(
                manifest._matches_component_key(
                    "model.layers.0.linear_attn.A_log",
                    "linear_attn.A_log",
                )
            )
            self.assertFalse(
                manifest._matches_component_key(
                    "model.layers.0.self_attn.A_log.weight",
                    "linear_attn.A_log",
                )
            )
            self.assertFalse(manifest._is_supported_text_tensor("model.unknown.weight", config))
            self.assertEqual(
                manifest._processing_sort_key("model.embed_tokens.weight", config),
                (0, -1, -1, 0),
            )
            self.assertEqual(
                manifest._processing_sort_key("model.unknown.weight", config),
                (3, -1, -1, 0),
            )
            self.assertEqual(
                manifest._processing_sort_key("model.layers.0.unknown.weight", config),
                (2, 0, len(config.arch_info.component_order), 0),
            )
            self.assertTrue(
                manifest._should_skip_tensor(
                    "model.layers.0.self_attn.k_proj.weight_scale",
                    tie_word_embeddings=False,
                )
            )
            self.assertTrue(
                manifest._should_skip_tensor(
                    "lm_head.weight",
                    tie_word_embeddings=True,
                )
            )
            self.assertTrue(manifest._is_language_model_only_skip("model.visual.patch_embed.proj.weight"))
            self.assertTrue(manifest._is_language_model_only_skip("mtp.fc.weight"))

            invalid_index_dir = root / "invalid-index"
            invalid_index_dir.mkdir()
            (invalid_index_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid sharded safetensors index"):
                manifest.get_sharded_files(invalid_index_dir)

            with self.assertRaisesRegex(FileNotFoundError, "No model.safetensors"):
                manifest.get_sharded_files(root / "missing-model")

            with patch("trillim.quantize._manifest.Path.is_file", return_value=True):
                self.assertTrue(manifest.resolve_quantize_binary().name.startswith("trillim-quantize"))
            with patch("trillim.quantize._manifest.Path.is_file", return_value=False):
                with self.assertRaisesRegex(FileNotFoundError, "Bundled quantizer binary not found"):
                    manifest.resolve_quantize_binary()

    def test_manifest_validation_adapter_helpers_and_quantizer_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = _write_llama_model(root)
            config = entrypoint.load_model_config(model_dir)

            visual_root = root / "visual-root"
            visual_root.mkdir()
            visual_model_dir = _write_llama_model(visual_root)
            visual_tensor_path = visual_model_dir / "model.safetensors"
            _write_safetensors(
                visual_tensor_path,
                {
                    "model.embed_tokens.weight": ("F16", (4, 250)),
                    "model.visual.patch_embed.proj.weight": ("F16", (1,)),
                },
            )
            with self.assertRaisesRegex(ValueError, "layer unsupported at this time"):
                manifest._validate_supported_model_tensors(
                    visual_model_dir,
                    entrypoint.load_model_config(visual_model_dir),
                    language_model_only=False,
                )

            qwen_dir = _write_qwen_multimodal_model(root)
            qwen_config = entrypoint.load_model_config(qwen_dir)
            with self.assertRaisesRegex(ValueError, "text-only quantization"):
                manifest._validate_supported_model_tensors(
                    qwen_dir,
                    qwen_config,
                    language_model_only=False,
                )
            manifest._validate_supported_model_tensors(
                qwen_dir,
                qwen_config,
                language_model_only=True,
            )

            missing_weights_model = _write_config_only_model(root, name="missing-weights")
            with patch("trillim.quantize._manifest._validate_supported_model_tensors"):
                with self.assertRaises(FileNotFoundError):
                    manifest.build_manifest(
                        missing_weights_model,
                        entrypoint.load_model_config(missing_weights_model),
                        output_dir=missing_weights_model,
                    )

            adapter_only_model = _write_config_only_model(
                root,
                name="adapter-only-model",
                payload={
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_size": 250,
                    "intermediate_size": 300,
                    "num_attention_heads": 5,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 5,
                    "vocab_size": 64,
                },
            )
            adapter_only_root = root / "adapter-only-root"
            adapter_only_root.mkdir()
            adapter_dir = _write_adapter(adapter_only_root)
            with patch("trillim.quantize._manifest._validate_supported_model_tensors"):
                adapter_manifest = manifest.build_manifest(
                    adapter_only_model,
                    entrypoint.load_model_config(adapter_only_model),
                    output_dir=adapter_dir,
                    adapter_dir=adapter_dir,
                    language_model_only=False,
                )
            adapter_payload = _read_manifest(adapter_manifest)
            self.assertEqual(adapter_payload["tensors"], [])
            self.assertEqual(adapter_payload["sections"], [])
            self.assertIsNotNone(adapter_payload["lora"])

            with patch("trillim.quantize._manifest._validate_supported_model_tensors"), patch(
                "trillim.quantize._manifest._ordered_text_tensors",
                return_value=[],
            ):
                empty_manifest_path = manifest.build_manifest(
                    model_dir,
                    config,
                    output_dir=root,
                    language_model_only=False,
                )
            self.assertEqual(_read_manifest(empty_manifest_path)["sections"], [])

            adapter_rslora_root = root / "adapter-rslora-root"
            adapter_rslora_root.mkdir()
            adapter_with_rslora = _write_adapter(
                adapter_rslora_root,
                target_modules=["q_proj"],
                tensors={
                    "model.layers.0.self_attn.q_proj.lora_A.weight": ("F32", (4, 250)),
                    "model.layers.0.self_attn.q_proj.lora_B.weight": ("F32", (250, 4)),
                },
            )
            adapter_config_path = adapter_with_rslora / "adapter_config.json"
            adapter_payload_json = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            adapter_payload_json["use_rslora"] = True
            adapter_payload_json["lora_alpha"] = 10
            adapter_config_path.write_text(json.dumps(adapter_payload_json), encoding="utf-8")
            entries, scale = manifest._build_lora_entries(
                adapter_with_rslora,
                config,
                [],
                {},
                {},
                {},
            )
            self.assertAlmostEqual(scale, 5.0)
            self.assertIsNotNone(entries[0][manifest.LORA_TARGETS.index("self_attn.q_proj")])
            adapter_header, adapter_data_start = manifest._get_header_and_offsets(
                adapter_with_rslora / "adapter_model.safetensors"
            )
            entries, scale = manifest._build_lora_entries(
                adapter_with_rslora,
                config,
                [adapter_with_rslora / "adapter_model.safetensors"],
                {adapter_with_rslora / "adapter_model.safetensors": 0},
                {adapter_with_rslora / "adapter_model.safetensors": adapter_header},
                {adapter_with_rslora / "adapter_model.safetensors": adapter_data_start},
            )
            self.assertAlmostEqual(scale, 5.0)
            self.assertEqual(entries[0][manifest.LORA_TARGETS.index("self_attn.q_proj")]["a_shard_idx"], 0)

            header = {
                "model.layers.0.self_attn.q_proj.lora_A.weight": {},
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": {},
            }
            self.assertEqual(
                manifest._find_lora_key(header, 0, "self_attn.q_proj", "A"),
                "model.layers.0.self_attn.q_proj.lora_A.weight",
            )
            self.assertEqual(
                manifest._find_lora_key(header, 0, "self_attn.q_proj", "B"),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            )
            self.assertEqual(
                manifest._parse_adapter_tensor_key(
                    "model.layers.12.self_attn.q_proj.lora_A.weight"
                ),
                (12, "self_attn.q_proj", "A"),
            )
            self.assertIsNone(
                manifest._parse_adapter_tensor_key(
                    "model.layers.not-a-layer.self_attn.q_proj.lora_A.weight"
                )
            )
            self.assertIsNone(manifest._parse_adapter_tensor_key("garbage"))
            self.assertEqual(
                manifest._expected_lora_dims(config, "mlp.down_proj"),
                (300, 250),
            )
            for invalid_value in (True, "bad", 0):
                with self.subTest(invalid_value=invalid_value):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        manifest._require_positive_int(invalid_value, "rank")

            bad_target_dir = root / "bad-target"
            bad_target_dir.mkdir()
            (bad_target_dir / "adapter_config.json").write_text(
                json.dumps({"r": 4, "target_modules": "q_proj"}),
                encoding="utf-8",
            )
            _write_safetensors(
                bad_target_dir / "adapter_model.safetensors",
                {
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
                        "F32",
                        (4, 250),
                    ),
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (
                        "F32",
                        (250, 4),
                    ),
                },
            )
            with self.assertRaisesRegex(ValueError, "target_modules must be a list"):
                manifest._read_adapter_metadata(bad_target_dir, config)

            missing_weights_adapter_dir = root / "missing-weights-adapter"
            missing_weights_adapter_dir.mkdir()
            (missing_weights_adapter_dir / "adapter_config.json").write_text(
                json.dumps({"r": 4, "target_modules": ["q_proj"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "adapter_model.safetensors"):
                manifest._read_adapter_metadata(missing_weights_adapter_dir, config)

            too_many_layers_dir = root / "too-many-layers"
            too_many_layers_dir.mkdir()
            (too_many_layers_dir / "adapter_config.json").write_text(
                json.dumps({"r": 4, "target_modules": ["q_proj"]}),
                encoding="utf-8",
            )
            _write_safetensors(
                too_many_layers_dir / "adapter_model.safetensors",
                {
                    "base_model.model.model.layers.9.self_attn.q_proj.lora_A.weight": (
                        "F32",
                        (4, 250),
                    ),
                    "base_model.model.model.layers.9.self_attn.q_proj.lora_B.weight": (
                        "F32",
                        (250, 4),
                    ),
                },
            )
            with self.assertRaisesRegex(ValueError, "only has 1 layers"):
                manifest._read_adapter_metadata(too_many_layers_dir, config)

            unsupported_key_dir = root / "unsupported-key-adapter"
            unsupported_key_dir.mkdir()
            (unsupported_key_dir / "adapter_config.json").write_text(
                json.dumps({"r": 4, "target_modules": ["q_proj"]}),
                encoding="utf-8",
            )
            bad_header = {
                "__metadata__": {"format": "pt"},
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                    "dtype": "F32",
                    "shape": [4, 250],
                    "data_offsets": [0, 4000],
                },
                "weird": {
                    "dtype": "F32",
                    "shape": [250, 4],
                    "data_offsets": [4000, 8000],
                },
            }
            bad_header_bytes = json.dumps(bad_header, separators=(",", ":")).encode("utf-8")
            (unsupported_key_dir / "adapter_model.safetensors").write_bytes(
                struct.pack("<Q", len(bad_header_bytes)) + bad_header_bytes + (b"\0" * 8000)
            )
            with self.assertRaisesRegex(ValueError, "layer unsupported at this time: weird"):
                manifest._read_adapter_metadata(unsupported_key_dir, config)

            with patch(
                "trillim.quantize._manifest._parse_adapter_tensor_key",
                return_value=(0, "unsupported.target", "A"),
            ):
                with self.assertRaisesRegex(ValueError, "layer unsupported at this time"):
                    manifest._read_adapter_metadata(unsupported_key_dir, config)

            with self.assertRaisesRegex(FileNotFoundError, "adapter_config.json"):
                manifest._read_adapter_config_file(root / "missing-adapter")

            weird_scale_file = root / "weird-scale.safetensors"
            weird_scale_header = {"__metadata__": {"format": "pt"}}
            weird_scale_bytes = json.dumps(weird_scale_header, separators=(",", ":")).encode("utf-8")
            weird_scale_file.write_bytes(
                struct.pack("<Q", len(weird_scale_bytes)) + weird_scale_bytes
            )

            class _WeightMap(dict):
                def __bool__(self):
                    return True

                def values(self):
                    return [model_dir / "model.safetensors"]

                def get(self, key, default=None):
                    if key.endswith("_scale"):
                        return weird_scale_file
                    return super().get(key, default)

            real_get_header_and_offsets = manifest._get_header_and_offsets

            def fake_get_header_and_offsets(shard_path: Path):
                if shard_path == weird_scale_file:
                    return {}, 8
                return real_get_header_and_offsets(shard_path)

            with patch("trillim.quantize._manifest.get_sharded_files", return_value=([model_dir / "model.safetensors"], _WeightMap())), patch(
                "trillim.quantize._manifest._validate_supported_model_tensors",
            ), patch(
                "trillim.quantize._manifest._get_header_and_offsets",
                side_effect=fake_get_header_and_offsets,
            ):
                weird_manifest_path = manifest.build_manifest(model_dir, config, output_dir=root)
            weird_tensor = next(
                tensor
                for tensor in _read_manifest(weird_manifest_path)["tensors"]
                if tensor["action"] == manifest.ACTION_REPACK_TERNARY
            )
            self.assertEqual(weird_tensor["has_scale"], 0)

            extra_shard = root / "extra-shard.safetensors"
            extra_shard.write_bytes(struct.pack("<Q", 2) + b"{}")

            class _ExpandedWeightMap(dict):
                def __bool__(self):
                    return True

                def values(self):
                    return [model_dir / "model.safetensors", extra_shard]

            with patch(
                "trillim.quantize._manifest.get_sharded_files",
                return_value=([model_dir / "model.safetensors"], _ExpandedWeightMap()),
            ), patch(
                "trillim.quantize._manifest._validate_supported_model_tensors",
            ):
                expanded_manifest = manifest.build_manifest(model_dir, config, output_dir=root)
            self.assertIn(str(extra_shard), _read_manifest(expanded_manifest)["shards"])

            command_output_dir = root / "quantizer-output"
            command_output_dir.mkdir()
            manifest_path = command_output_dir / ".quantize_manifest.bin"
            manifest_path.write_bytes(b"manifest")
            temp_path = command_output_dir / "qmodel.tensors.tmp"
            temp_path.write_bytes(b"temp")
            tied_config = replace(entrypoint.load_model_config(model_dir), tie_word_embeddings=True)

            recorded_commands: list[list[str]] = []

            def fake_run(command, check):
                del check
                recorded_commands.append(list(command))
                return types.SimpleNamespace(returncode=0)

            with patch("trillim.quantize._manifest.build_manifest", return_value=manifest_path), patch(
                "trillim.quantize._manifest.subprocess.run",
                side_effect=fake_run,
            ):
                manifest.run_model_quantizer(
                    Path("/tmp/binary"),
                    model_dir,
                    tied_config,
                    output_dir=command_output_dir,
                    language_model_only=False,
                )
            self.assertIn("--tie-embeddings", recorded_commands[0])
            self.assertFalse(manifest_path.exists())
            self.assertFalse(temp_path.exists())

            manifest_path.write_bytes(b"manifest")
            temp_path.write_bytes(b"temp")
            with patch("trillim.quantize._manifest.build_manifest", return_value=manifest_path), patch(
                "trillim.quantize._manifest.subprocess.run",
                return_value=types.SimpleNamespace(returncode=7),
            ):
                with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
                    manifest.run_model_quantizer(
                        Path("/tmp/binary"),
                        model_dir,
                        config,
                        output_dir=command_output_dir,
                        language_model_only=False,
                    )
            self.assertFalse(manifest_path.exists())
            self.assertFalse(temp_path.exists())

            adapter_output_dir = root / "adapter-quantizer-output"
            adapter_output_dir.mkdir()
            manifest_path = adapter_output_dir / ".quantize_manifest.bin"
            manifest_path.write_bytes(b"manifest")
            unused = adapter_output_dir / ".unused-qmodel.tensors"
            unused.write_bytes(b"temp")
            unused_tmp = adapter_output_dir / ".unused-qmodel.tensors.tmp"
            unused_tmp.write_bytes(b"temp")
            command_adapter_root = root / "command-adapter"
            command_adapter_root.mkdir()
            with patch("trillim.quantize._manifest.build_manifest", return_value=manifest_path), patch(
                "trillim.quantize._manifest._read_adapter_config_file",
                return_value={"r": 8},
            ), patch(
                "trillim.quantize._manifest.subprocess.run",
                side_effect=fake_run,
            ):
                manifest.run_adapter_quantizer(
                    Path("/tmp/binary"),
                    model_dir,
                    config,
                    adapter_dir=_write_adapter(command_adapter_root),
                    output_dir=adapter_output_dir,
                    language_model_only=False,
                )
            self.assertIn("--lora-rank", recorded_commands[-1])
            self.assertFalse(manifest_path.exists())
            self.assertFalse(unused.exists())
            self.assertFalse(unused_tmp.exists())

            manifest_path.write_bytes(b"manifest")
            unused.write_bytes(b"temp")
            unused_tmp.write_bytes(b"temp")
            with patch("trillim.quantize._manifest.build_manifest", return_value=manifest_path), patch(
                "trillim.quantize._manifest._read_adapter_config_file",
                return_value={"r": 8},
            ), patch(
                "trillim.quantize._manifest.subprocess.run",
                return_value=types.SimpleNamespace(returncode=9),
            ):
                with self.assertRaisesRegex(RuntimeError, "exited with code 9"):
                    manifest.run_adapter_quantizer(
                        Path("/tmp/binary"),
                        model_dir,
                        tied_config,
                        adapter_dir=adapter_with_rslora,
                        output_dir=adapter_output_dir,
                        language_model_only=False,
                    )
            self.assertFalse(manifest_path.exists())
            self.assertFalse(unused.exists())
            self.assertFalse(unused_tmp.exists())

            extra_path = root / "extra.txt"
            extra_path.write_text("temp", encoding="utf-8")
            manifest._cleanup_paths(extra_path, root / "missing.txt")
            self.assertFalse(extra_path.exists())

    def test_output_staging_recovery_remote_code_and_entrypoint_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patched_model_store() as store_root:
                source_dir = store_root / "source"
                source_dir.mkdir()
                preferred = output.prepare_output_target(source_dir)
                preferred.parent.mkdir(parents=True, exist_ok=True)
                preferred.mkdir()
                (preferred.parent / f"{preferred.name}-2").mkdir()
                (preferred.parent / f"{preferred.name}-3").mkdir()
                with patch("trillim.quantize._output.sys_stdin_isatty", return_value=True), patch(
                    "trillim.quantize._output.sys_stdout_isatty",
                    return_value=True,
                ), patch("builtins.input", side_effect=["n"]):
                    deduped = output.prepare_output_target(source_dir)
                self.assertEqual(deduped.name, "source-TRNQ-4")

                staging = preferred.parent / f"{preferred.name}-new"
                staging.mkdir()
                with self.assertRaisesRegex(RuntimeError, "Staging directory is still present"):
                    output.build_staging_dir(preferred)
                shutil_ready = preferred.parent / "publish-target"
                staging = preferred.parent / "publish-target-new"
                with self.assertRaisesRegex(RuntimeError, "Staging directory not found"):
                    output.publish_staging_dir(shutil_ready)
                staging.mkdir()
                with self.assertRaisesRegex(RuntimeError, "Staging directory is incomplete"):
                    output.publish_staging_dir(shutil_ready)
                output.mark_staging_complete(staging)
                output.publish_staging_dir(shutil_ready)
                self.assertTrue(shutil_ready.is_dir())

                existing_target = preferred.parent / "existing-target"
                existing_target.mkdir()
                (existing_target / "old.txt").write_text("old", encoding="utf-8")
                staging = preferred.parent / "existing-target-new"
                staging.mkdir()
                (staging / "new.txt").write_text("new", encoding="utf-8")
                output.mark_staging_complete(staging)
                output.publish_staging_dir(existing_target)
                self.assertTrue((existing_target / "new.txt").is_file())
                self.assertFalse((preferred.parent / "existing-target-old").exists())

                for name in ("bad-staging-new", "bad-backup-old", "bad-target"):
                    bad_path = preferred.parent / name
                    bad_path.write_text("not-a-dir", encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "not a directory"):
                        output.recover_publish_state(
                            preferred.parent / name.removesuffix("-new").removesuffix("-old")
                        )
                    bad_path.unlink()

                recovery_target = preferred.parent / "recover-me"
                recovery_target.mkdir()
                recovery_staging = preferred.parent / "recover-me-new"
                recovery_backup = preferred.parent / "recover-me-old"
                recovery_staging.mkdir()
                recovery_backup.mkdir()
                output.recover_publish_state(recovery_target)
                self.assertFalse(recovery_staging.exists())
                self.assertFalse(recovery_backup.exists())

                promoted_target = preferred.parent / "promote-me"
                promoted_staging = preferred.parent / "promote-me-new"
                promoted_backup = preferred.parent / "promote-me-old"
                promoted_staging.mkdir()
                output.mark_staging_complete(promoted_staging)
                promoted_backup.mkdir()
                output.recover_publish_state(promoted_target)
                self.assertTrue(promoted_target.exists())
                self.assertFalse(promoted_backup.exists())

                restored_target = preferred.parent / "restore-me"
                restored_staging = preferred.parent / "restore-me-new"
                restored_backup = preferred.parent / "restore-me-old"
                restored_staging.mkdir()
                restored_backup.mkdir()
                (restored_backup / "old.txt").write_text("old", encoding="utf-8")
                output.recover_publish_state(restored_target)
                self.assertTrue((restored_target / "old.txt").is_file())
                self.assertFalse(restored_staging.exists())

                direct_restore_target = preferred.parent / "direct-restore"
                direct_restore_backup = preferred.parent / "direct-restore-old"
                direct_restore_backup.mkdir()
                (direct_restore_backup / "old.txt").write_text("old", encoding="utf-8")
                output.recover_publish_state(direct_restore_target)
                self.assertTrue((direct_restore_target / "old.txt").is_file())

                cleanup_target = preferred.parent / "cleanup-me"
                cleanup_staging = preferred.parent / "cleanup-me-new"
                cleanup_staging.mkdir()
                output.recover_publish_state(cleanup_target)
                self.assertFalse(cleanup_staging.exists())

                promote_target = preferred.parent / "promote-direct"
                promote_staging = preferred.parent / "promote-direct-new"
                promote_staging.mkdir()
                output.mark_staging_complete(promote_staging)
                output.recover_publish_state(promote_target)
                self.assertTrue(promote_target.is_dir())

                broken_target = preferred.parent / "broken-publish"
                broken_staging = preferred.parent / "broken-publish-new"
                broken_staging.mkdir()
                output.mark_staging_complete(broken_staging)
                with patch(
                    "trillim.quantize._output.os.replace",
                    side_effect=OSError("boom"),
                ):
                    with self.assertRaises(OSError):
                        output.publish_staging_dir(broken_target)

                output_json = preferred.parent / "payload.json"
                output._write_json(output_json, {"x": 1})
                self.assertTrue(output_json.read_text(encoding="utf-8").endswith("\n"))

                nested_source = preferred.parent / "nested" / "source.txt"
                nested_source.parent.mkdir(parents=True, exist_ok=True)
                nested_source.write_text("copy", encoding="utf-8")
                nested_destination = preferred.parent / "deep" / "copy.txt"
                output._copy_file(nested_source, nested_destination)
                self.assertEqual(nested_destination.read_text(encoding="utf-8"), "copy")

                self.assertTrue(output._should_skip_adapter_path(Path("__pycache__/x.pyc")))
                self.assertTrue(output._should_skip_adapter_path(Path("weights.tmp")))
                self.assertTrue(output._should_skip_adapter_path(Path("adapter_model.safetensors")))
                self.assertTrue(output._should_skip_adapter_path(Path("adapter_model.safetensors.index.json")))
                self.assertTrue(output._should_skip_adapter_path(Path("adapter_model.bin")))
                self.assertFalse(output._should_skip_adapter_path(Path("nested/keep.txt")))

            adapter_dir = root / "adapter-copy"
            adapter_dir.mkdir()
            (adapter_dir / "__pycache__").mkdir()
            (adapter_dir / "__pycache__" / "cached.pyc").write_bytes(b"x")
            (adapter_dir / "weights.tmp").write_text("skip", encoding="utf-8")
            (adapter_dir / "adapter_model.bin").write_text("skip", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_text("skip", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors.index.json").write_text("skip", encoding="utf-8")
            (adapter_dir / "nested").mkdir()
            (adapter_dir / "nested" / "keep.txt").write_text("keep", encoding="utf-8")
            adapter_output = root / "adapter-out"
            output.copy_adapter_support_files(adapter_dir, adapter_output)
            self.assertTrue((adapter_output / "nested" / "keep.txt").is_file())
            self.assertFalse((adapter_output / "adapter_model.bin").exists())

            empty_model_dir = _write_config_only_model(root, name="empty-model")
            empty_output = root / "empty-output"
            empty_output.mkdir()
            output.write_adapter_metadata(
                empty_output,
                config=entrypoint.load_model_config(empty_model_dir),
                adapter_dir=root / "missing-adapter-config",
                model_dir=empty_model_dir,
            )
            adapter_metadata = json.loads(
                (empty_output / "trillim_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(adapter_metadata["source_model"], "")

            self.assertFalse(output._load_optional_json(root / "missing.json"))
            payload_path = root / "not-dict.json"
            payload_path.write_text("[]", encoding="utf-8")
            self.assertIsNone(output._load_optional_json(payload_path))

            refs = output._collect_remote_code_class_refs(
                {
                    "config": {
                        "auto_map": {
                            "AutoTokenizer": ["tokenizer_mod.Tokenizer", "tokenizer_mod.Tokenizer"],
                            "AutoConfig": "config_mod.Config",
                        }
                    },
                    "tokenizer_config": {
                        "auto_map": ["tokenizer_mod.Tokenizer", None],
                    },
                }
            )
            self.assertEqual(refs, ["tokenizer_mod.Tokenizer", "config_mod.Config"])
            self.assertEqual(
                output._extract_auto_map_refs({"auto_map": ["a.A", "", None]}, key="AutoTokenizer"),
                ["a.A"],
            )
            self.assertEqual(
                output._extract_auto_map_refs({"auto_map": {"AutoConfig": ["x.Y", None]}}, key="AutoConfig"),
                ["x.Y"],
            )
            self.assertEqual(output._extract_auto_map_refs(None, key="AutoTokenizer"), [])
            self.assertIsNone(
                output._build_bundle_tokenizer_config(
                    root,
                    tokenizer_config={
                        "auto_map": {"AutoTokenizer": "tokenizer_mod.Tokenizer"},
                        "tokenizer_class": "Tokenizer",
                    },
                )
            )
            self.assertIsNone(
                output._build_bundle_tokenizer_config(
                    root,
                    tokenizer_config={"tokenizer_class": ""},
                )
            )
            self.assertEqual(
                output._parse_remote_code_module_path("tokenizer_mod.Tokenizer"),
                Path("tokenizer_mod.py"),
            )
            with self.assertRaisesRegex(ValueError, "External remote-code repositories"):
                output._parse_remote_code_module_path("repo--tokenizer_mod.Tokenizer")
            with self.assertRaisesRegex(ValueError, "currently unsupported"):
                output._parse_remote_code_module_path("TokenizerOnly")
            with self.assertRaisesRegex(ValueError, "Package-scoped remote-code entry points"):
                output._parse_remote_code_module_path("pkg.tokenizer_mod.Tokenizer")

            no_match_model = root / "bundle-no-match"
            no_match_model.mkdir()
            (no_match_model / "helper.py").write_text("class Other: pass\n", encoding="utf-8")
            self.assertIsNone(
                output._find_local_class_module(no_match_model, class_name="MissingTokenizer")
            )

            preferred_module_model = root / "bundle-preferred"
            preferred_module_model.mkdir()
            (preferred_module_model / "alpha.py").write_text("class Tokenizer: pass\n", encoding="utf-8")
            (preferred_module_model / "tokenization_alpha.py").write_text(
                "class Tokenizer: pass\n",
                encoding="utf-8",
            )
            self.assertEqual(
                output._find_local_class_module(preferred_module_model, class_name="Tokenizer"),
                Path("tokenization_alpha.py"),
            )

            ambiguous_module_model = root / "bundle-ambiguous"
            ambiguous_module_model.mkdir()
            (ambiguous_module_model / "alpha.py").write_text(
                "class Tokenizer: pass\n",
                encoding="utf-8",
            )
            (ambiguous_module_model / "beta.py").write_text(
                "class Tokenizer: pass\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "defined in multiple local modules"):
                output._find_local_class_module(ambiguous_module_model, class_name="Tokenizer")

            self.assertEqual(
                output._collect_bundle_support_class_refs(
                    {
                        "config": {
                            "auto_map": {"AutoTokenizer": "tokenizer_mod.Tokenizer"},
                        },
                        "tokenizer_config": {
                            "auto_map": {"AutoTokenizer": ["tokenizer_mod.Tokenizer", None]},
                        },
                    }
                ),
                ["tokenizer_mod.Tokenizer"],
            )

            module_path = root / "imports.py"
            module_path.write_text(
                "from .sibling import thing\nfrom . import cousin\nfrom . import cousin\n",
                encoding="utf-8",
            )
            self.assertEqual(
                output._relative_import_module_names(module_path),
                ["sibling", "cousin"],
            )
            module_path.write_text("from ..parent import thing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Parent relative imports"):
                output._relative_import_module_names(module_path)
            module_path.write_text("from .pkg.module import thing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Package-scoped relative imports"):
                output._relative_import_module_names(module_path)
            module_path.write_text("from pkg import thing\n", encoding="utf-8")
            self.assertEqual(output._relative_import_module_names(module_path), [])
            module_path.write_text("from . import *\n", encoding="utf-8")
            self.assertEqual(output._relative_import_module_names(module_path), [])

            model_dir = _write_remote_code_model(
                root,
                name="remote-code",
                config_payload={
                    "auto_map": {"AutoConfig": "config_mod.Config"},
                    "text_config": {"auto_map": {"AutoTokenizer": "tokenizer_mod.Tokenizer"}},
                },
                tokenizer_payload={"auto_map": ["tokenizer_mod.Tokenizer"]},
                files={
                    "config_mod.py": "from .shared import value\nclass Config: pass\n",
                    "tokenizer_mod.py": "from .shared import value\nclass Tokenizer: pass\n",
                    "shared.py": "value = 1\n",
                },
            )
            self.assertTrue(output._has_remote_code_references(model_dir))
            self.assertEqual(
                sorted(str(path) for path in output._collect_remote_code_files(model_dir)),
                ["config_mod.py", "shared.py", "tokenizer_mod.py"],
            )

            missing_remote_model = _write_remote_code_model(
                root,
                name="remote-missing",
                config_payload={"auto_map": {"AutoConfig": "missing_mod.Config"}},
                files={},
            )
            with self.assertRaisesRegex(ValueError, "Remote-code module not found"):
                output._collect_remote_code_files(missing_remote_model)

            deep_remote_model = _write_remote_code_model(
                root,
                name="remote-deep",
                config_payload={"auto_map": {"AutoConfig": "config_mod.Config"}},
                files={
                    "config_mod.py": "from .shared import value\nclass Config: pass\n",
                    "shared.py": "from .leaf import value\n",
                    "leaf.py": "value = 1\n",
                },
            )
            with patch("trillim.quantize._output._MAX_REMOTE_CODE_DEPTH", 0):
                with self.assertRaisesRegex(ValueError, "supported depth"):
                    output._collect_remote_code_files(deep_remote_model)

            wide_remote_model = _write_remote_code_model(
                root,
                name="remote-wide",
                config_payload={"auto_map": {"AutoConfig": "config_mod.Config"}},
                files={
                    "config_mod.py": "from .one import a\nfrom .two import b\nclass Config: pass\n",
                    "one.py": "a = 1\n",
                    "two.py": "b = 2\n",
                },
            )
            with patch("trillim.quantize._output._MAX_REMOTE_CODE_FILES", 2):
                with self.assertRaisesRegex(ValueError, "supported file budget"):
                    output._collect_remote_code_files(wide_remote_model)

            large_remote_model = _write_remote_code_model(
                root,
                name="remote-large",
                config_payload={"auto_map": {"AutoConfig": "config_mod.Config"}},
                files={"config_mod.py": "x = '" + ("y" * 64) + "'\nclass Config: pass\n"},
            )
            with patch("trillim.quantize._output._MAX_REMOTE_CODE_BYTES", 32):
                with self.assertRaisesRegex(ValueError, "supported byte budget"):
                    output._collect_remote_code_files(large_remote_model)

            package_remote_model = _write_remote_code_model(
                root,
                name="remote-package",
                config_payload={"auto_map": {"AutoConfig": "config_mod.Config"}},
                files={
                    "config_mod.py": "from .pkg import value\nclass Config: pass\n",
                    "pkg/__init__.py": "value = 1\n",
                },
            )
            with self.assertRaisesRegex(ValueError, "Package-scoped relative imports"):
                output._collect_remote_code_files(package_remote_model)

            resolved_import = output._resolve_relative_import_module_path(
                source_relative_path=Path("tokenizer_mod.py"),
                module_name="shared",
                model_dir=model_dir,
            )
            self.assertEqual(resolved_import, Path("shared.py"))

            model_dir_with_pkg = _write_remote_code_model(
                root,
                name="remote-package-init",
                config_payload={"auto_map": {"AutoConfig": "config_mod.Config"}},
                files={
                    "config_mod.py": "class Config: pass\n",
                    "pkg/__init__.py": "value = 1\n",
                },
            )
            with self.assertRaisesRegex(ValueError, "Package-scoped relative imports"):
                output._resolve_relative_import_module_path(
                    source_relative_path=Path("config_mod.py"),
                    module_name="pkg",
                    model_dir=model_dir_with_pkg,
                )

            with patch(
                "trillim.quantize._output.importlib_metadata.version",
                side_effect=output.importlib_metadata.PackageNotFoundError,
            ):
                pyproject_payload = tomllib.loads(
                    (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    output._project_version(),
                    str(pyproject_payload["project"]["version"]),
                )
            with patch(
                "trillim.quantize._output.importlib_metadata.version",
                return_value="9.9.9",
            ):
                self.assertEqual(output._project_version(), "9.9.9")

            real_stdin = os.sys.stdin
            real_stdout = os.sys.stdout
            try:
                os.sys.stdin = object()
                os.sys.stdout = object()
                self.assertFalse(output.sys_stdin_isatty())
                self.assertFalse(output.sys_stdout_isatty())
            finally:
                os.sys.stdin = real_stdin
                os.sys.stdout = real_stdout

            with patch("builtins.input", return_value="no"):
                self.assertFalse(output._confirm_overwrite(Path("bundle")))

            with patched_model_store() as store_root:
                inside_store = store_root / "Local" / "managed"
                inside_store.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "must not be inside"):
                    entrypoint._normalize_source_dir(inside_store, label="Model directory")

            file_path = root / "not-a-dir.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "is not a directory"):
                entrypoint._normalize_source_dir(file_path, label="Model directory")

            outside_dir = root / "outside"
            outside_dir.mkdir()
            self.assertEqual(
                entrypoint._normalize_source_dir(outside_dir, label="Model directory"),
                outside_dir.resolve(),
            )

            with patch(
                "trillim.quantize._entrypoint._normalize_source_dir",
                side_effect=[Path("/tmp/model"), Path("/tmp/model")],
            ):
                with self.assertRaisesRegex(ValueError, "must be different"):
                    entrypoint.quantize("model", "adapter")

            entrypoint_model_root = root / "entrypoint-model-root"
            entrypoint_model_root.mkdir()
            model_config = entrypoint.load_model_config(_write_llama_model(entrypoint_model_root))
            with patch(
                "trillim.quantize._entrypoint._normalize_source_dir",
                return_value=Path("/tmp/model"),
            ), patch(
                "trillim.quantize._entrypoint.load_model_config",
                return_value=model_config,
            ), patch(
                "trillim.quantize._entrypoint.determine_language_model_only",
                return_value=False,
            ), patch(
                "trillim.quantize._entrypoint.resolve_quantize_binary",
                return_value=Path("/tmp/binary"),
            ), patch(
                "trillim.quantize._entrypoint.prepare_output_target",
                return_value=Path("/tmp/target"),
            ), patch(
                "trillim.quantize._entrypoint.build_staging_dir",
                return_value=Path("/tmp/staging"),
            ), patch(
                "trillim.quantize._entrypoint.run_model_quantizer",
            ) as run_model_quantizer, patch(
                "trillim.quantize._entrypoint.copy_model_support_files",
            ) as copy_model_support_files, patch(
                "trillim.quantize._entrypoint.write_model_metadata",
            ) as write_model_metadata, patch(
                "trillim.quantize._entrypoint.mark_staging_complete",
            ) as mark_complete, patch(
                "trillim.quantize._entrypoint.publish_staging_dir",
            ) as publish_staging_dir:
                result = entrypoint.quantize("/tmp/model")
            self.assertEqual(result.bundle_type, "model")
            run_model_quantizer.assert_called_once()
            copy_model_support_files.assert_called_once()
            write_model_metadata.assert_called_once()
            mark_complete.assert_called_once()
            publish_staging_dir.assert_called_once()

            with patch(
                "trillim.quantize._entrypoint._normalize_source_dir",
                side_effect=[Path("/tmp/model"), Path("/tmp/adapter")],
            ), patch(
                "trillim.quantize._entrypoint.load_model_config",
                return_value=model_config,
            ), patch(
                "trillim.quantize._entrypoint.determine_language_model_only",
                return_value=True,
            ), patch(
                "trillim.quantize._entrypoint.resolve_quantize_binary",
                return_value=Path("/tmp/binary"),
            ), patch(
                "trillim.quantize._entrypoint.validate_adapter_source",
            ) as validate_adapter_source, patch(
                "trillim.quantize._entrypoint.prepare_output_target",
                return_value=Path("/tmp/adapter-target"),
            ), patch(
                "trillim.quantize._entrypoint.build_staging_dir",
                return_value=Path("/tmp/adapter-staging"),
            ), patch(
                "trillim.quantize._entrypoint.run_adapter_quantizer",
            ) as run_adapter_quantizer, patch(
                "trillim.quantize._entrypoint.copy_adapter_support_files",
            ) as copy_adapter_support_files, patch(
                "trillim.quantize._entrypoint.write_adapter_metadata",
            ) as write_adapter_metadata, patch(
                "trillim.quantize._entrypoint.mark_staging_complete",
            ) as mark_complete, patch(
                "trillim.quantize._entrypoint.publish_staging_dir",
            ) as publish_staging_dir, redirect_stdout(io.StringIO()) as output_text:
                result = entrypoint.quantize("/tmp/model", "/tmp/adapter")
            self.assertEqual(result.bundle_type, "adapter")
            self.assertTrue(result.used_language_model_only)
            self.assertIn("text inference", output_text.getvalue())
            validate_adapter_source.assert_called_once()
            run_adapter_quantizer.assert_called_once()
            copy_adapter_support_files.assert_called_once()
            write_adapter_metadata.assert_called_once()
            mark_complete.assert_called_once()
            publish_staging_dir.assert_called_once()

    def test_copy_adapter_support_files_strips_implicit_tokenizer_loader_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": 5,
                        "tokenizer_class": "TokenizersBackend",
                        "auto_map": {"AutoConfig": "config_mod.Config"},
                    }
                ),
                encoding="utf-8",
            )
            (adapter_dir / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "TokenizersBackend",
                        "chat_template": "{{ messages }}",
                        "pad_token": "<pad>",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "adapter-out"

            output.copy_adapter_support_files(adapter_dir, output_dir)

            copied_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            copied_tokenizer_config = json.loads(
                (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                copied_config,
                {
                    "eos_token_id": 5,
                    "auto_map": {"AutoConfig": "config_mod.Config"},
                },
            )
            self.assertEqual(
                copied_tokenizer_config,
                {
                    "chat_template": "{{ messages }}",
                    "pad_token": "<pad>",
                },
            )

    def test_copy_adapter_support_files_preserves_explicit_auto_tokenizer_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": 5,
                        "tokenizer_class": "AdapterTokenizer",
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None],
                            "AutoConfig": "config_mod.Config",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (adapter_dir / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "AdapterTokenizer",
                        "auto_map": ["tokenization_adapter.AdapterTokenizer", None],
                        "chat_template": "{{ messages }}",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "adapter-out"

            output.copy_adapter_support_files(adapter_dir, output_dir)

            copied_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            copied_tokenizer_config = json.loads(
                (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                copied_config,
                {
                    "eos_token_id": 5,
                    "tokenizer_class": "AdapterTokenizer",
                    "auto_map": {
                        "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None],
                        "AutoConfig": "config_mod.Config",
                    },
                },
            )
            self.assertEqual(
                copied_tokenizer_config,
                {
                    "tokenizer_class": "AdapterTokenizer",
                    "auto_map": ["tokenization_adapter.AdapterTokenizer", None],
                    "chat_template": "{{ messages }}",
                },
            )

    def test_copy_adapter_support_files_copies_non_object_tokenizer_config_verbatim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "tokenizer_config.json").write_text("[]", encoding="utf-8")
            output_dir = root / "adapter-out"

            output.copy_adapter_support_files(adapter_dir, output_dir)

            self.assertEqual(
                (output_dir / "tokenizer_config.json").read_text(encoding="utf-8"),
                "[]",
            )

    def test_sanitize_adapter_tokenizer_loader_fields_removes_empty_auto_map_shapes(self):
        self.assertEqual(
            output._sanitize_adapter_tokenizer_loader_fields(
                {
                    "tokenizer_class": "AdapterTokenizer",
                    "auto_map": {"AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None]},
                },
                adapter_has_explicit_auto_tokenizer=False,
            ),
            {},
        )
        self.assertEqual(
            output._sanitize_adapter_tokenizer_loader_fields(
                {
                    "tokenizer_class": "AdapterTokenizer",
                    "auto_map": ["tokenization_adapter.AdapterTokenizer", None],
                },
                adapter_has_explicit_auto_tokenizer=False,
            ),
            {},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
