# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for quantization helpers, manifest generation, and CLI flow."""

from __future__ import annotations

import json
import math
import os
import runpy
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from trillim.model_arch import ARCH_REGISTRY, ArchType, LORA_TARGETS, ModelConfig
import trillim.quantize as quantize


def _read_manifest(path: str) -> dict:
    with open(path, "rb") as handle:
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

        remainder = handle.read()
        if not remainder:
            return {"shards": shards, "tensors": tensors, "lora": None}

    offset = 0

    def _read(fmt: str):
        nonlocal offset
        size = struct.calcsize(fmt)
        value = struct.unpack(fmt, remainder[offset:offset + size])[0]
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
        "lora": {
            "num_layers": num_layers,
            "targets_per_layer": targets_per_layer,
            "scale": scale,
            "layers": layers,
        },
    }


class QuantizeTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "arch_type": ArchType.LLAMA,
            "arch_info": ARCH_REGISTRY["llamaforcausallm"],
            "hidden_dim": 256,
            "intermediate_dim": 384,
            "hidden_dim_orig": 250,
            "intermediate_dim_orig": 300,
            "num_layers": 1,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 32,
            "norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _write_model(self, root: str, *, sharded: bool = False) -> str:
        model_dir = Path(root) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "_name_or_path": "Org/BaseModel",
                    "rope_theta": 7000.0,
                    "max_position_embeddings": 2048,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        tensors = {
            "model.embed_tokens.weight": np.zeros((4, 250), dtype=np.float16),
            "model.layers.0.input_layernorm.weight": np.zeros((250,), dtype=np.float16),
            "model.layers.0.self_attn.q_proj.weight": np.zeros((250, 250), dtype=np.float16),
            "model.layers.0.self_attn.k_proj.weight": np.zeros((250, 250), dtype=np.int8),
            "model.layers.0.self_attn.k_proj.weight_scale": np.zeros((250,), dtype=np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.zeros((300, 250), dtype=np.float16),
            "model.layers.0.mlp.down_proj.bias": np.zeros((250,), dtype=np.float16),
            "model.norm.weight": np.zeros((250,), dtype=np.float16),
            "lm_head.weight": np.zeros((4, 250), dtype=np.float16),
            "model.layers.0.self_attn.rotary_emb.inv_freq": np.zeros((8,), dtype=np.float32),
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
            shard2_tensors = {
                key: value for key, value in tensors.items() if key not in shard1_tensors
            }
            save_file(shard1_tensors, str(shard1))
            save_file(shard2_tensors, str(shard2))
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
            save_file(tensors, str(model_dir / "model.safetensors"))

        return str(model_dir)

    def _write_qwen_model(self, root: str) -> str:
        model_dir = Path(root) / "qwen-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "_name_or_path": "Org/Qwen3.5-4B",
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "rope_theta": 123.0,
                    "max_position_embeddings": 456,
                    "text_config": {
                        "hidden_size": 2560,
                        "intermediate_size": 9216,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 16,
                        "num_key_value_heads": 4,
                        "head_dim": 160,
                        "vocab_size": 248320,
                        "max_position_embeddings": 262144,
                        "rms_norm_eps": 1e-6,
                        "rope_theta": 10000000.0,
                        "hidden_act": "silu",
                        "eos_token_id": 248044,
                        "tie_word_embeddings": True,
                        "attention_bias": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        save_file(
            {
                "model.embed_tokens.weight": np.zeros((16, 2560), dtype=np.float16),
                "model.layers.0.input_layernorm.weight": np.zeros((2560,), dtype=np.float16),
                "model.layers.0.self_attn.q_proj.weight": np.zeros((2560, 2560), dtype=np.float16),
                "model.layers.0.self_attn.k_proj.weight": np.zeros((640, 2560), dtype=np.float16),
                "model.layers.0.self_attn.v_proj.weight": np.zeros((640, 2560), dtype=np.float16),
                "model.layers.0.self_attn.o_proj.weight": np.zeros((2560, 2560), dtype=np.float16),
                "model.layers.0.mlp.gate_proj.weight": np.zeros((9216, 2560), dtype=np.float16),
                "model.layers.0.mlp.up_proj.weight": np.zeros((9216, 2560), dtype=np.float16),
                "model.layers.0.mlp.down_proj.weight": np.zeros((2560, 9216), dtype=np.float16),
                "model.norm.weight": np.zeros((2560,), dtype=np.float16),
                "model.layers.0.self_attn.rotary_emb.inv_freq": np.zeros((80,), dtype=np.float32),
            },
            str(model_dir / "model.safetensors"),
        )
        return str(model_dir)

    def _write_adapter(
        self,
        root: str,
        *,
        rank: int = 4,
        use_rslora: bool = False,
        q_proj_a_shape: tuple[int, int] = (4, 250),
        q_proj_b_shape: tuple[int, int] = (250, 4),
        gate_proj_b_shape: tuple[int, int] = (300, 4),
        extra_layer: bool = False,
    ) -> str:
        adapter_dir = Path(root) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": rank,
                    "lora_alpha": rank * 2,
                    "target_modules": ["q_proj", "gate_proj", "ignored"],
                    "use_rslora": use_rslora,
                    "base_model_name_or_path": "Org/BaseModel",
                }
            ),
            encoding="utf-8",
        )
        tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.zeros(q_proj_a_shape, dtype=np.float32),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.zeros(q_proj_b_shape, dtype=np.float32),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": np.zeros((rank, 250), dtype=np.float32),
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": np.zeros(gate_proj_b_shape, dtype=np.float32),
        }
        if extra_layer:
            tensors["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"] = np.zeros((rank, 250), dtype=np.float32)
            tensors["base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight"] = np.zeros((250, rank), dtype=np.float32)
        save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
        return str(adapter_dir)

    def test_metadata_shard_discovery_and_tensor_name_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_model(temp_dir)
            metadata = quantize.get_tensor_metadata(os.path.join(model_dir, "model.safetensors"))
            self.assertTrue(any(item["key"] == "model.embed_tokens.weight" for item in metadata))

            meta_path = Path(temp_dir) / "meta.safetensors"
            save_file(
                {"tensor": np.zeros((1,), dtype=np.float32)},
                str(meta_path),
                metadata={"info": "present"},
            )
            self.assertEqual(quantize.get_tensor_metadata(str(meta_path)), [{"key": "tensor", "start": 0, "shape": [1]}])

            shards, weight_map = quantize.get_sharded_files(model_dir)
            self.assertEqual(shards, [os.path.join(model_dir, "model.safetensors")])
            self.assertEqual(weight_map, {})
            self.assertIn("model.norm.weight", quantize.get_all_tensor_names(model_dir))

            sharded_model = self._write_model(temp_dir + "_sharded", sharded=True)
            shards, weight_map = quantize.get_sharded_files(sharded_model)
            self.assertEqual(len(shards), 2)
            self.assertIn("model.layers.0.self_attn.q_proj.weight", weight_map)
            self.assertIn("lm_head.weight", quantize.get_all_tensor_names(sharded_model))

            with self.assertRaisesRegex(FileNotFoundError, "No model.safetensors"):
                quantize.get_sharded_files(temp_dir)

    def test_processing_order_and_dtype_helpers_cover_known_branches(self):
        ordered = quantize.get_processing_order(
            [
                {"key": "model.layers.0.mlp.down_proj.bias"},
                {"key": "model.norm.weight"},
                {"key": "model.embed_tokens.weight"},
                {"key": "model.layers.0.self_attn.q_proj.weight"},
                {"key": "model.layers.0.unknown.weight"},
                {"key": "lm_head.weight"},
                {"key": "misc.weight"},
            ],
            self._config().arch_info,
        )
        self.assertEqual(
            [item["key"] for item in ordered],
            [
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.norm.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.down_proj.bias",
                "model.layers.0.unknown.weight",
                "misc.weight",
            ],
        )

        self.assertEqual(quantize._safetensors_dtype_code("F32"), (quantize.DTYPE_F32, 4))
        self.assertEqual(quantize._safetensors_dtype_code("F16"), (quantize.DTYPE_F16, 2))
        self.assertEqual(quantize._safetensors_dtype_code("BF16"), (quantize.DTYPE_BF16, 2))
        self.assertEqual(quantize._safetensors_dtype_code("I8"), (quantize.DTYPE_I8, 1))
        self.assertEqual(quantize._safetensors_dtype_code("U8"), (quantize.DTYPE_U8, 1))
        with self.assertRaisesRegex(ValueError, "Unknown safetensors dtype"):
            quantize._safetensors_dtype_code("BAD")

    def test_header_parsing_and_manifest_generation_cover_actions_padding_and_scales(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_model(temp_dir)
            header, data_start = quantize._get_header_and_offsets(os.path.join(model_dir, "model.safetensors"))
            self.assertIn("model.embed_tokens.weight", header)
            self.assertGreater(data_start, 8)

            manifest_path = quantize.write_manifest(model_dir, self._config())
            manifest = _read_manifest(manifest_path)

            self.assertEqual(len(manifest["shards"]), 1)
            self.assertEqual(len(manifest["tensors"]), 8)

            embedding_entry = manifest["tensors"][0]
            self.assertEqual(embedding_entry["action"], quantize.ACTION_BF16_RAW)

            repack_entry = next(entry for entry in manifest["tensors"] if entry["has_scale"] == 1)
            self.assertEqual(repack_entry["action"], quantize.ACTION_REPACK_TERNARY)
            self.assertGreater(repack_entry["scale_size"], 0)

            padded_entry = next(
                entry for entry in manifest["tensors"] if entry["row"] == 300 and entry["col"] == 250
            )
            self.assertEqual((padded_entry["padded_row"], padded_entry["padded_col"]), (384, 256))

            tied_manifest_path = quantize.write_manifest(
                model_dir,
                self._config(tie_word_embeddings=True),
                manifest_dir=temp_dir,
            )
            tied_manifest = _read_manifest(tied_manifest_path)
            self.assertEqual(len(tied_manifest["tensors"]), 7)

    def test_write_manifest_supports_adapter_only_mode_and_sharded_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sharded_model = self._write_model(temp_dir, sharded=True)
            manifest = _read_manifest(quantize.write_manifest(sharded_model, self._config()))
            self.assertEqual(len(manifest["shards"]), 2)

            empty_model = Path(temp_dir) / "empty-model"
            empty_model.mkdir()
            adapter_dir = self._write_adapter(temp_dir + "_adapter")
            adapter_manifest = _read_manifest(
                quantize.write_manifest(str(empty_model), self._config(), adapter_dir=adapter_dir)
            )
            self.assertEqual(adapter_manifest["tensors"], [])
            self.assertEqual(adapter_manifest["lora"]["num_layers"], 1)

            skipped_manifest = _read_manifest(
                quantize.write_manifest(str(empty_model), self._config(), adapter_dir=adapter_dir, skip_model=True)
            )
            self.assertEqual(skipped_manifest["tensors"], [])

            with self.assertRaisesRegex(FileNotFoundError, "No model.safetensors"):
                quantize.write_manifest(str(empty_model), self._config())

    def test_write_manifest_supports_qwen35_model_config_and_tensor_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_qwen_model(temp_dir)
            config = ModelConfig.from_config_json(
                os.path.join(model_dir, "config.json"),
                model_dir=model_dir,
            )

            manifest = _read_manifest(quantize.write_manifest(model_dir, config))

        self.assertEqual(config.arch_type, ArchType.QWEN35)
        self.assertEqual(config.rope_theta, 10000000.0)
        self.assertEqual(config.max_position_embeddings, 262144)
        self.assertTrue(config.tie_word_embeddings)
        self.assertEqual(len(manifest["tensors"]), 10)

        embedding_entry = manifest["tensors"][0]
        norm_entry = manifest["tensors"][1]
        q_proj_entry = next(
            entry for entry in manifest["tensors"]
            if entry["row"] == 2560 and entry["col"] == 2560 and entry["action"] == quantize.ACTION_TERNARY_QUANTIZE
        )
        k_proj_entry = next(
            entry for entry in manifest["tensors"]
            if entry["row"] == 640 and entry["col"] == 2560
        )

        self.assertEqual(embedding_entry["action"], quantize.ACTION_BF16_RAW)
        self.assertEqual(norm_entry["action"], quantize.ACTION_BF16_RAW)
        self.assertEqual(q_proj_entry["action"], quantize.ACTION_TERNARY_QUANTIZE)
        self.assertEqual((q_proj_entry["row"], q_proj_entry["col"]), (2560, 2560))
        self.assertEqual((k_proj_entry["row"], k_proj_entry["col"]), (640, 2560))

    def test_write_manifest_handles_inconsistent_shard_maps_in_mocked_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shard = Path(temp_dir) / "shard.safetensors"
            extra_shard = Path(temp_dir) / "extra.safetensors"
            scale_shard = Path(temp_dir) / "scale.safetensors"
            save_file(
                {"model.layers.0.self_attn.k_proj.weight": np.zeros((250, 250), dtype=np.int8)},
                str(shard),
            )
            save_file({"unused.weight": np.zeros((1,), dtype=np.float32)}, str(extra_shard))
            save_file(
                {"model.layers.0.self_attn.k_proj.weight_scale": np.zeros((250,), dtype=np.float32)},
                str(scale_shard),
            )

            class _WeirdWeightMap(dict):
                def get(self, key, default=None):
                    if key == "model.layers.0.self_attn.k_proj.weight_scale":
                        return str(scale_shard)
                    return super().get(key, default)

            with (
                patch(
                    "trillim.quantize.get_sharded_files",
                    return_value=(
                        [str(shard)],
                        _WeirdWeightMap({"model.layers.0.self_attn.k_proj.weight": str(extra_shard)}),
                    ),
                ),
                patch("trillim.quantize.get_tensor_metadata", return_value=[{"key": "model.layers.0.self_attn.k_proj.weight", "shape": [250, 250], "start": 0}]),
            ):
                manifest = _read_manifest(quantize.write_manifest(temp_dir, self._config()))

            self.assertEqual(len(manifest["shards"]), 3)
            self.assertEqual(manifest["tensors"][0]["has_scale"], 1)

    def test_build_lora_entries_maps_short_targets_and_rslora_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = self._write_adapter(temp_dir, use_rslora=True)
            shard_paths: list[str] = []
            shard_idx_map: dict[str, int] = {}
            shard_headers: dict[str, dict] = {}
            shard_data_starts: dict[str, int] = {}

            entries, scale = quantize._build_lora_entries(
                adapter_dir,
                self._config(),
                shard_paths,
                shard_idx_map,
                shard_headers,
                shard_data_starts,
            )

            self.assertEqual(len(entries), 1)
            self.assertEqual(len(entries[0]), len(LORA_TARGETS))
            self.assertIsNone(entries[0][0])
            self.assertIsNotNone(entries[0][2])
            self.assertIsNotNone(entries[0][4])
            self.assertAlmostEqual(scale, 4.0)
            self.assertEqual(len(shard_paths), 1)

            with self.assertRaisesRegex(FileNotFoundError, "No adapter_config.json"):
                quantize._build_lora_entries(
                    temp_dir,
                    self._config(),
                    [],
                    {},
                    {},
                    {},
                )

    def test_find_binary_and_cpp_quantizer_build_command_and_cleanup(self):
        with patch.dict(
            "sys.modules",
            {"trillim._bin_path": SimpleNamespace(quantize_bin=lambda: "/fake/bin")},
        ):
            self.assertEqual(quantize._find_quantize_binary(), "/fake/bin")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_model(temp_dir)
            adapter_dir = self._write_adapter(temp_dir + "_adapter")
            model_output_dir = str(Path(temp_dir) / "model-out")
            adapter_output_dir = str(Path(temp_dir) / "adapter-out")
            Path(model_output_dir).mkdir()
            Path(adapter_output_dir).mkdir()
            config = self._config(tie_word_embeddings=True)

            seen_cmds: list[list[str]] = []

            def fake_manifest(*args, **kwargs):
                manifest_path = Path(model_output_dir) / ".quantize_manifest.bin"
                manifest_path.write_bytes(b"manifest")
                return str(manifest_path)

            def fake_run(cmd, capture_output=False):
                seen_cmds.append(cmd)
                (Path(model_output_dir) / "qmodel.tensors.tmp").write_bytes(b"tmp")
                return SimpleNamespace(returncode=0)

            with (
                patch("trillim.quantize.write_manifest", side_effect=fake_manifest),
                patch("trillim.quantize.subprocess.run", side_effect=fake_run),
            ):
                quantize._run_cpp_quantizer(
                    "/fake/bin",
                    model_dir,
                    config,
                    model_output_dir,
                    adapter_dir=adapter_dir,
                    adapter_output_dir=adapter_output_dir,
                )

            self.assertIn("--tie-embeddings", seen_cmds[0])
            self.assertIn("--lora-output", seen_cmds[0])
            self.assertFalse((Path(model_output_dir) / ".quantize_manifest.bin").exists())
            self.assertFalse((Path(model_output_dir) / "qmodel.tensors.tmp").exists())

            rope_theta_index = seen_cmds[0].index("--rope-theta")
            max_pos_index = seen_cmds[0].index("--max-pos")
            self.assertEqual(seen_cmds[0][rope_theta_index + 1], str(config.rope_theta))
            self.assertEqual(seen_cmds[0][max_pos_index + 1], str(config.max_position_embeddings))

            with self.assertRaisesRegex(ValueError, "model_output_dir and model_dir resolve to the same path"):
                quantize._run_cpp_quantizer("/fake/bin", model_dir, config, model_dir)

            with self.assertRaisesRegex(ValueError, "adapter_output_dir is required"):
                quantize._run_cpp_quantizer(
                    "/fake/bin",
                    model_dir,
                    config,
                    model_output_dir,
                    adapter_dir=adapter_dir,
                )

            with self.assertRaisesRegex(ValueError, "adapter_output_dir and adapter_dir resolve to the same path"):
                quantize._run_cpp_quantizer(
                    "/fake/bin",
                    model_dir,
                    config,
                    model_output_dir,
                    adapter_dir=adapter_dir,
                    adapter_output_dir=adapter_dir,
                )

            with (
                patch("trillim.quantize.write_manifest", side_effect=fake_manifest),
                patch("trillim.quantize.subprocess.run", return_value=SimpleNamespace(returncode=7)),
            ):
                with self.assertRaisesRegex(RuntimeError, "C\\+\\+ quantizer exited with code 7"):
                    quantize._run_cpp_quantizer("/fake/bin", model_dir, config, model_output_dir)

    def test_run_cpp_quantizer_uses_normalized_rope_fields_instead_of_raw_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "rope_theta": 123.0,
                        "max_position_embeddings": 456,
                        "text_config": {
                            "rope_theta": 10000000.0,
                            "max_position_embeddings": 262144,
                        },
                    }
                ),
                encoding="utf-8",
            )
            model_output_dir = str(Path(temp_dir) / "model-out")
            Path(model_output_dir).mkdir()
            config = self._config(
                arch_type=ArchType.QWEN35,
                arch_info=ARCH_REGISTRY["qwen3_5forconditionalgeneration"],
                rope_theta=10000000.0,
                max_position_embeddings=262144,
            )
            seen_cmds: list[list[str]] = []

            def fake_manifest(*args, **kwargs):
                manifest_path = Path(model_output_dir) / ".quantize_manifest.bin"
                manifest_path.write_bytes(b"manifest")
                return str(manifest_path)

            def fake_run(cmd, capture_output=False):
                seen_cmds.append(cmd)
                return SimpleNamespace(returncode=0)

            with (
                patch("trillim.quantize.write_manifest", side_effect=fake_manifest),
                patch("trillim.quantize.subprocess.run", side_effect=fake_run),
            ):
                quantize._run_cpp_quantizer(
                    "/fake/bin",
                    str(model_dir),
                    config,
                    model_output_dir,
                )

            rope_theta_index = seen_cmds[0].index("--rope-theta")
            max_pos_index = seen_cmds[0].index("--max-pos")
            self.assertEqual(seen_cmds[0][rope_theta_index + 1], "10000000.0")
            self.assertEqual(seen_cmds[0][max_pos_index + 1], "262144")

    def test_run_cpp_lora_only_and_file_copy_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_model(temp_dir)
            adapter_dir = self._write_adapter(temp_dir + "_adapter")
            adapter_output_dir = str(Path(temp_dir) / "adapter-out")
            Path(adapter_output_dir).mkdir()
            config = self._config(tie_word_embeddings=True)
            seen_cmds: list[list[str]] = []

            def fake_manifest(*args, **kwargs):
                manifest_path = Path(adapter_output_dir) / ".quantize_manifest.bin"
                manifest_path.write_bytes(b"manifest")
                return str(manifest_path)

            def fake_run(cmd, capture_output=False):
                seen_cmds.append(cmd)
                return SimpleNamespace(returncode=0)

            with (
                patch("trillim.quantize.write_manifest", side_effect=fake_manifest),
                patch("trillim.quantize.subprocess.run", side_effect=fake_run),
            ):
                quantize._run_cpp_lora_only(
                    "/fake/bin",
                    model_dir,
                    config,
                    adapter_dir,
                    adapter_output_dir,
                )

            self.assertIn("--lora-output", seen_cmds[0])
            self.assertFalse((Path(adapter_output_dir) / ".quantize_manifest.bin").exists())

            with (
                patch("trillim.quantize.write_manifest", side_effect=fake_manifest),
                patch("trillim.quantize.subprocess.run", return_value=SimpleNamespace(returncode=3)),
            ):
                with self.assertRaisesRegex(RuntimeError, "C\\+\\+ quantizer exited with code 3"):
                    quantize._run_cpp_lora_only(
                        "/fake/bin",
                        model_dir,
                        config,
                        adapter_dir,
                        adapter_output_dir,
                    )

            with self.assertRaisesRegex(ValueError, "adapter_output_dir and adapter_dir resolve to the same path"):
                quantize._run_cpp_lora_only("/fake/bin", model_dir, config, adapter_dir, adapter_dir)

            adapter_dir_path = Path(adapter_dir)
            (adapter_dir_path / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "chat_template": "{{ messages }}",
                        "bos_token": "<s>",
                        "eos_token": "</s>",
                        "pad_token": "<pad>",
                        "unk_token": "<unk>",
                        "extra": "ignored",
                    }
                ),
                encoding="utf-8",
            )
            (adapter_dir_path / "chat_template.jinja").write_text("{{ adapter }}", encoding="utf-8")
            (adapter_dir_path / "tokenizer.json").write_text("{}", encoding="utf-8")
            quantize._copy_adapter_tokenizer_files(adapter_dir, adapter_output_dir)
            self.assertTrue((Path(adapter_output_dir) / "lora_tokenizer_config.json").exists())
            self.assertTrue((Path(adapter_output_dir) / "lora_chat_template.jinja").exists())
            self.assertTrue((Path(adapter_output_dir) / "lora_tokenizer.json").exists())

            self.assertTrue(quantize._make_adapter_output_dir(adapter_dir).endswith("-TRNQ"))
            self.assertTrue(quantize._make_model_output_dir(model_dir).endswith("-TRNQ"))

    def test_copy_model_files_validate_dims_and_config_writers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = self._write_model(temp_dir)
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            (Path(model_dir) / "module").mkdir()
            (Path(model_dir) / "module" / "code.py").write_text("x = 1", encoding="utf-8")
            (Path(model_dir) / "rope.cache").write_bytes(b"skip")
            (Path(model_dir) / "qmodel.tensors").write_bytes(b"skip")
            (Path(model_dir) / ".quantize_manifest.bin").write_bytes(b"skip")
            (Path(model_dir) / "weights.safetensors").write_bytes(b"skip")
            existing_dir = output_dir / "module"
            existing_dir.mkdir()
            (existing_dir / "old.py").write_text("old", encoding="utf-8")

            quantize._copy_model_files(model_dir, str(output_dir))
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "module" / "code.py").exists())
            self.assertFalse((output_dir / "weights.safetensors").exists())

            with tempfile.TemporaryDirectory() as adapter_root:
                config = self._config()
                adapter_dir = Path(adapter_root) / "adapter"
                adapter_dir.mkdir()

                quantize._validate_adapter_dims(str(adapter_dir), config)

                (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 4}), encoding="utf-8")
                quantize._validate_adapter_dims(str(adapter_dir), config)

                adapter_path = self._write_adapter(
                    adapter_root + "_good",
                    rank=4,
                    q_proj_a_shape=(4, 250),
                    q_proj_b_shape=(250, 4),
                    gate_proj_b_shape=(300, 4),
                )
                quantize._validate_adapter_dims(adapter_path, config)

                with self.assertRaisesRegex(ValueError, "weights for 2 layers"):
                    quantize._validate_adapter_dims(self._write_adapter(adapter_root + "_layers", extra_layer=True), config)
                with self.assertRaisesRegex(ValueError, r"lora_A.*r=4"):
                    quantize._validate_adapter_dims(
                        self._write_adapter(adapter_root + "_a", q_proj_a_shape=(5, 250)),
                        config,
                    )
                with self.assertRaisesRegex(ValueError, r"lora_B.*r=4"):
                    quantize._validate_adapter_dims(
                        self._write_adapter(adapter_root + "_b", q_proj_b_shape=(250, 5)),
                        config,
                    )
                with self.assertRaisesRegex(ValueError, "hidden_size is 250"):
                    quantize._validate_adapter_dims(
                        self._write_adapter(adapter_root + "_hidden", q_proj_a_shape=(4, 128)),
                        config,
                    )
                with self.assertRaisesRegex(ValueError, "intermediate_size is 300"):
                    quantize._validate_adapter_dims(
                        self._write_adapter(adapter_root + "_intermediate", gate_proj_b_shape=(128, 4)),
                        config,
                    )

                missing_qproj_dir = Path(adapter_root) / "missing-qproj"
                missing_qproj_dir.mkdir()
                (missing_qproj_dir / "adapter_config.json").write_text(json.dumps({"r": 4}), encoding="utf-8")
                save_file(
                    {
                        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": np.zeros((4, 250), dtype=np.float32),
                        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": np.zeros((250, 4), dtype=np.float32),
                    },
                    str(missing_qproj_dir / "adapter_model.safetensors"),
                )
                quantize._validate_adapter_dims(str(missing_qproj_dir), config)

            with patch("trillim.quantize.compute_base_model_hash", return_value="hash-value"):
                quantize._write_trillim_model_config(str(output_dir), self._config(), model_dir)
                adapter_dir = self._write_adapter(temp_dir + "_cfg")
                quantize._write_trillim_adapter_config(str(output_dir), self._config(), adapter_dir, model_dir)

            model_cfg = json.loads((output_dir / "trillim_config.json").read_text(encoding="utf-8"))
            self.assertEqual(model_cfg["base_model_config_hash"], "hash-value")
            self.assertEqual(model_cfg["architecture"], "llama")

    def test_main_covers_argument_validation_and_model_and_adapter_flows(self):
        with patch.object(__import__("sys"), "argv", ["trillim quantize", "model"]):
            with self.assertRaisesRegex(ValueError, "specify --model and/or --adapter"):
                quantize.main()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            with patch.object(__import__("sys"), "argv", ["trillim quantize", str(model_dir), "--model"]):
                with self.assertRaisesRegex(FileNotFoundError, "config.json not found"):
                    quantize.main()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            config = self._config()

            with (
                patch.object(__import__("sys"), "argv", ["trillim quantize", str(model_dir), "--model", "--adapter", str(adapter_dir)]),
                patch("trillim.quantize._find_quantize_binary", return_value="/fake/bin"),
                patch("trillim.quantize.ModelConfig.from_config_json", return_value=config),
                patch("trillim.quantize._make_model_output_dir", return_value="/tmp/model-out"),
                patch("trillim.quantize._make_adapter_output_dir", return_value="/tmp/adapter-out"),
                patch("trillim.quantize._validate_adapter_dims") as validate_mock,
                patch("trillim.quantize._run_cpp_quantizer") as run_quantizer_mock,
                patch("trillim.quantize._copy_model_files") as copy_model_mock,
                patch("trillim.quantize._write_trillim_model_config") as write_model_cfg_mock,
                patch("trillim.quantize._copy_adapter_tokenizer_files") as copy_adapter_mock,
                patch("trillim.quantize._write_trillim_adapter_config") as write_adapter_cfg_mock,
                patch("builtins.print"),
            ):
                quantize.main()

            validate_mock.assert_called_once_with(str(adapter_dir), config)
            run_quantizer_mock.assert_called_once()
            copy_model_mock.assert_called_once_with(str(model_dir), "/tmp/model-out")
            write_model_cfg_mock.assert_called_once_with("/tmp/model-out", config, str(model_dir))
            copy_adapter_mock.assert_called_once_with(str(adapter_dir), "/tmp/adapter-out")
            write_adapter_cfg_mock.assert_called_once_with("/tmp/adapter-out", config, str(adapter_dir), str(model_dir))

            with (
                patch.object(__import__("sys"), "argv", ["trillim quantize", str(model_dir), "--model"]),
                patch("trillim.quantize._find_quantize_binary", return_value="/fake/bin"),
                patch("trillim.quantize.ModelConfig.from_config_json", return_value=config),
                patch("trillim.quantize._make_model_output_dir", return_value="/tmp/model-only"),
                patch("trillim.quantize._run_cpp_quantizer"),
                patch("trillim.quantize._copy_model_files"),
                patch("trillim.quantize._write_trillim_model_config"),
                patch("builtins.print") as print_mock,
            ):
                quantize.main()

            self.assertIn("Usage: trillim chat /tmp/model-only", [call.args[0] for call in print_mock.call_args_list])

            with (
                patch.object(__import__("sys"), "argv", ["trillim quantize", str(model_dir), "--adapter", str(adapter_dir)]),
                patch("trillim.quantize._find_quantize_binary", return_value="/fake/bin"),
                patch("trillim.quantize.ModelConfig.from_config_json", return_value=config),
                patch("trillim.quantize._make_adapter_output_dir", return_value="/tmp/adapter-out"),
                patch("trillim.quantize._validate_adapter_dims"),
                patch("trillim.quantize._run_cpp_lora_only") as run_lora_only_mock,
                patch("trillim.quantize._copy_adapter_tokenizer_files"),
                patch("trillim.quantize._write_trillim_adapter_config"),
                patch("builtins.print"),
                ):
                    quantize.main()

            run_lora_only_mock.assert_called_once()

            with patch.object(__import__("sys"), "argv", ["trillim quantize", str(model_dir), "--adapter", str(temp_dir) + "/missing"]):
                with (
                    patch("trillim.quantize._find_quantize_binary", return_value="/fake/bin"),
                    patch("trillim.quantize.ModelConfig.from_config_json", return_value=config),
                    self.assertRaisesRegex(FileNotFoundError, "Adapter directory not found"),
                ):
                    quantize.main()

    def test_quantize_module_runs_main_when_executed_as_script(self):
        with patch.object(__import__("sys"), "argv", ["trillim quantize", "model"]):
            with self.assertRaisesRegex(ValueError, "specify --model and/or --adapter"):
                runpy.run_path(quantize.__file__, run_name="__main__")

if __name__ == "__main__":
    unittest.main()
