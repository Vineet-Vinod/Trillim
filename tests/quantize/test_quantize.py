"""Tests for local quantization helpers and entrypoints."""

from __future__ import annotations

import io
import json
import math
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from trillim._bundle_metadata import (
    CURRENT_FORMAT_VERSION,
    compute_base_model_config_hash,
)
from trillim.quantize import quantize
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
            self.assertTrue((model_output_dir / "configuration_trillim.py").is_file())
            self.assertTrue((model_output_dir / "shared.py").is_file())
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

            with patch("trillim.quantize._entrypoint.run_model_quantizer", side_effect=fake_run_model_quantizer):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
