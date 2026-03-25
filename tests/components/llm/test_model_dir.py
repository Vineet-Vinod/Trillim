"""Tests for model-directory validation."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.components.llm._config import ActivationType, ArchitectureType, InitConfig
from trillim.components.llm._model_dir import (
    prepare_runtime_files,
    validate_lora_dir,
    validate_model_dir,
)
from trillim.errors import ModelValidationError
from tests.components.llm.support import model_dir, write_adapter_bundle


class ModelDirectoryTests(unittest.TestCase):
    def test_validate_model_dir_rejects_missing_model_directory(self):
        with model_dir() as root:
            missing = root.parent / "missing-model-dir"
        with self.assertRaisesRegex(ModelValidationError, "does not exist"):
            validate_model_dir(missing)

    def test_validate_model_dir_extracts_runtime_metadata(self):
        with model_dir() as root:
            info = validate_model_dir(root)

        self.assertEqual(info.name, root.name)
        self.assertEqual(info.arch_type, ArchitectureType.LLAMA)
        self.assertEqual(info.activation, ActivationType.SILU)
        self.assertEqual(info.eos_tokens, (2,))

    def test_validate_model_dir_rejects_missing_weights(self):
        with model_dir() as root:
            (root / "qmodel.tensors").unlink()
            with self.assertRaisesRegex(ModelValidationError, "qmodel.tensors"):
                validate_model_dir(root)

    def test_validate_model_dir_rejects_unsupported_architecture(self):
        with model_dir(architecture="UnsupportedForCausalLM") as root:
            with self.assertRaisesRegex(ModelValidationError, "Unsupported model architecture"):
                validate_model_dir(root)

    def test_validate_model_dir_handles_text_config_models(self):
        with model_dir(
            architecture="LlamaForCausalLM",
            text_config={
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "vocab_size": 256,
                "max_position_embeddings": 512,
                "hidden_act": "silu",
                "eos_token_id": 3,
            },
        ) as root:
            info = validate_model_dir(root)

        self.assertEqual(info.eos_tokens, (3, 2))

    def test_validate_model_dir_rejects_missing_rope_cache(self):
        with model_dir() as root:
            (root / "rope.cache").unlink()
            with self.assertRaisesRegex(ModelValidationError, "rope.cache"):
                validate_model_dir(root)

    def test_validate_model_dir_rejects_symlinked_model_directory(self):
        with model_dir() as root:
            symlink = root.parent / f"{root.name}-model-link"
            symlink.symlink_to(root, target_is_directory=True)

            with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                validate_model_dir(symlink)

    def test_validate_model_dir_rejects_symlinked_config_json(self):
        with model_dir() as root:
            replacement = root / "config-real.json"
            replacement.write_text((root / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "config.json").unlink()
            (root / "config.json").symlink_to(replacement)

            with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                validate_model_dir(root)

    def test_validate_model_dir_rejects_symlinked_required_artifacts(self):
        for filename, payload in (
            ("qmodel.tensors", b"quantized-model"),
            ("rope.cache", b"rope-cache"),
        ):
            with self.subTest(filename=filename):
                with model_dir() as root:
                    replacement = root / f"real-{filename}"
                    replacement.write_bytes(payload)
                    (root / filename).unlink()
                    (root / filename).symlink_to(replacement)

                    with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                        validate_model_dir(root)

    def test_validate_model_dir_rejects_symlinked_optional_tokenizer(self):
        with model_dir() as root:
            replacement = root / "tokenizer-real.json"
            replacement.write_text(
                (root / "tokenizer.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "tokenizer.json").unlink()
            (root / "tokenizer.json").symlink_to(replacement)

            with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                validate_model_dir(root)

    def test_validate_lora_dir_rejects_symlinked_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "qmodel.lora").write_bytes(b"adapter")
            symlink = root / "adapter-link"
            symlink.symlink_to(adapter, target_is_directory=True)

            with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                validate_lora_dir(symlink)

    def test_validate_lora_dir_requires_qmodel_lora(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            adapter.mkdir()

            with self.assertRaisesRegex(ModelValidationError, "qmodel.lora not found"):
                validate_lora_dir(adapter)

    def test_validate_lora_dir_rejects_symlinked_qmodel_lora(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "qmodel.lora"
            target.write_bytes(b"adapter")
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "qmodel.lora").symlink_to(target)

            with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                validate_lora_dir(adapter)

    def test_validate_lora_dir_requires_trillim_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            adapter.mkdir()
            (adapter / "qmodel.lora").write_bytes(b"adapter")

            with self.assertRaisesRegex(ModelValidationError, "trillim_config.json not found"):
                validate_lora_dir(adapter)

    def test_validate_lora_dir_requires_supported_compatibility_metadata(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            adapter.mkdir()
            (adapter / "qmodel.lora").write_bytes(b"adapter")
            (adapter / "trillim_config.json").write_text(
                json.dumps({"format_version": 3}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ModelValidationError, "missing or unsupported"):
                validate_lora_dir(adapter)
            with self.assertRaisesRegex(ModelValidationError, "missing or unsupported"):
                validate_lora_dir(adapter, model_dir=root)

    def test_validate_lora_dir_rejects_non_object_trillim_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            adapter.mkdir()
            (adapter / "qmodel.lora").write_bytes(b"adapter")
            (adapter / "trillim_config.json").write_text(
                "[]",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ModelValidationError, "must be a JSON object"):
                validate_lora_dir(adapter)

    def test_validate_lora_dir_rejects_incompatible_base_model(self):
        with model_dir() as root, model_dir(extra_config={"vocab_size": 1024}) as other_root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=other_root)

            with self.assertRaisesRegex(ModelValidationError, "Adapter/model mismatch"):
                validate_lora_dir(adapter, model_dir=root)

    def test_validate_lora_dir_accepts_equivalent_defaulted_kv_head_configs(self):
        with model_dir() as explicit_root, model_dir() as defaulted_root, tempfile.TemporaryDirectory() as temp_dir:
            config_path = defaulted_root / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.pop("num_key_value_heads")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=defaulted_root)

            resolved = validate_lora_dir(adapter, model_dir=explicit_root)

        self.assertEqual(resolved, adapter)

    def test_prepare_runtime_files_merges_lora_metadata_and_prefers_adapter_truth(self):
        with model_dir(
            extra_config={
                "rope_parameters": {"rope_theta": 10000.0, "base_only": 1},
                "tie_word_embeddings": False,
            }
        ) as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "vocab.txt").write_text("base-vocab\n", encoding="utf-8")
            (root / "merges.txt").write_text("base-merges\n", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"unused")
            (adapter / "added_tokens.json").write_text('{"adapter": 1}', encoding="utf-8")
            (adapter / "config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": 5,
                        "rope_parameters": {"rope_theta": 20000.0},
                        "tie_word_embeddings": True,
                    }
                ),
                encoding="utf-8",
            )
            (root / "generation_config.json").write_text(
                json.dumps({"top_k": 10, "max_new_tokens": 16}),
                encoding="utf-8",
            )
            (adapter / "generation_config.json").write_text(
                json.dumps({"temperature": 0.3, "top_k": 99}),
                encoding="utf-8",
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps({"source": "base"}),
                encoding="utf-8",
            )
            (adapter / "tokenizer_config.json").write_text(
                json.dumps({"source": "adapter"}),
                encoding="utf-8",
            )
            (root / "chat_template.jinja").write_text("base-template", encoding="utf-8")
            (adapter / "chat_template.jinja").write_text("adapter-template", encoding="utf-8")
            (adapter / "tokenizer.json").write_text(
                json.dumps({"added_tokens": [{"content": "</s>", "id": 5}]}),
                encoding="utf-8",
            )

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=root, lora_dir=adapter),
                trust_remote_code=False,
            )
            overlay_path = runtime_files.metadata_dir
            try:
                merged_config = json.loads((overlay_path / "config.json").read_text(encoding="utf-8"))
                merged_generation = json.loads(
                    (overlay_path / "generation_config.json").read_text(encoding="utf-8")
                )

                self.assertEqual(merged_config["rope_parameters"]["rope_theta"], 20000.0)
                self.assertEqual(merged_config["rope_parameters"]["base_only"], 1)
                self.assertTrue(merged_config["tie_word_embeddings"])
                self.assertEqual(merged_generation["top_k"], 99)
                self.assertEqual(merged_generation["temperature"], 0.3)
                self.assertEqual(merged_generation["max_new_tokens"], 16)
                self.assertEqual(
                    json.loads((overlay_path / "tokenizer_config.json").read_text(encoding="utf-8")),
                    {"source": "adapter"},
                )
                self.assertEqual(
                    (overlay_path / "chat_template.jinja").read_text(encoding="utf-8"),
                    "adapter-template",
                )
                self.assertEqual(
                    (overlay_path / "vocab.txt").read_text(encoding="utf-8"),
                    "base-vocab\n",
                )
                self.assertEqual(
                    (overlay_path / "merges.txt").read_text(encoding="utf-8"),
                    "base-merges\n",
                )
                self.assertEqual(
                    json.loads((overlay_path / "added_tokens.json").read_text(encoding="utf-8")),
                    {"adapter": 1},
                )
                self.assertTrue((overlay_path / "qmodel.tensors").is_file())
                self.assertTrue((overlay_path / "rope.cache").is_file())
                self.assertTrue((overlay_path / "qmodel.lora").is_file())
                self.assertTrue(os.path.samefile(overlay_path / "qmodel.tensors", root / "qmodel.tensors"))
                self.assertTrue(os.path.samefile(overlay_path / "rope.cache", root / "rope.cache"))
                self.assertTrue(os.path.samefile(overlay_path / "qmodel.lora", adapter / "qmodel.lora"))
                self.assertFalse((overlay_path / "model.safetensors").exists())

                info = validate_model_dir(root, metadata_dir=overlay_path)

                self.assertEqual(info.eos_tokens, (5,))
                self.assertEqual(info.rope_theta, 20000.0)
                self.assertTrue(info.tie_word_embeddings)
            finally:
                runtime_files.cleanup()

            self.assertFalse(overlay_path.exists())

    def test_prepare_runtime_files_preserves_adapter_overrides_for_text_config_models(self):
        with model_dir(
            architecture="LlamaForCausalLM",
            text_config={
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "num_key_value_heads": 4,
                "vocab_size": 256,
                "max_position_embeddings": 512,
                "hidden_act": "silu",
                "eos_token_id": 3,
                "rope_parameters": {"rope_theta": 10000.0},
                "tie_word_embeddings": False,
            },
        ) as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (adapter / "config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": 5,
                        "rope_parameters": {"rope_theta": 20000.0},
                        "tie_word_embeddings": True,
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "tokenizer.json").write_text(
                json.dumps({"added_tokens": [{"content": "</s>", "id": 5}]}),
                encoding="utf-8",
            )

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=root, lora_dir=adapter),
                trust_remote_code=False,
            )
            overlay_path = runtime_files.metadata_dir
            try:
                merged_config = json.loads((overlay_path / "config.json").read_text(encoding="utf-8"))

                self.assertEqual(merged_config["eos_token_id"], 5)
                self.assertEqual(
                    merged_config["rope_parameters"]["rope_theta"],
                    20000.0,
                )
                self.assertTrue(merged_config["tie_word_embeddings"])

                info = validate_model_dir(root, metadata_dir=overlay_path)

                self.assertEqual(info.eos_tokens, (5,))
                self.assertEqual(info.rope_theta, 20000.0)
                self.assertTrue(info.tie_word_embeddings)
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_preserves_adapter_text_config_over_flat_base(self):
        with model_dir(
            extra_config={
                "eos_token_id": 3,
                "rope_parameters": {"rope_theta": 10000.0},
                "tie_word_embeddings": False,
            }
        ) as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (adapter / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["LlamaForCausalLM"],
                        "text_config": {
                            "eos_token_id": 5,
                            "rope_parameters": {"rope_theta": 20000.0},
                            "tie_word_embeddings": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "tokenizer.json").write_text(
                json.dumps({"added_tokens": [{"content": "</s>", "id": 5}]}),
                encoding="utf-8",
            )

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=root, lora_dir=adapter),
                trust_remote_code=False,
            )
            overlay_path = runtime_files.metadata_dir
            try:
                merged_config = json.loads((overlay_path / "config.json").read_text(encoding="utf-8"))

                self.assertEqual(merged_config["eos_token_id"], 5)
                self.assertEqual(merged_config["rope_parameters"]["rope_theta"], 20000.0)
                self.assertTrue(merged_config["tie_word_embeddings"])

                info = validate_model_dir(root, metadata_dir=overlay_path)

                self.assertEqual(info.eos_tokens, (5,))
                self.assertEqual(info.rope_theta, 20000.0)
                self.assertTrue(info.tie_word_embeddings)
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_merges_partial_tokenizer_config_with_base_fallback(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "chat_template": "base-template",
                        "base_only": 1,
                        "nested": {"base": "keep"},
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "chat_template": "adapter-template",
                        "nested": {"adapter": "win"},
                    }
                ),
                encoding="utf-8",
            )

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=root, lora_dir=adapter),
                trust_remote_code=False,
            )
            overlay_path = runtime_files.metadata_dir
            try:
                merged = json.loads(
                    (overlay_path / "tokenizer_config.json").read_text(encoding="utf-8")
                )

                self.assertEqual(merged["chat_template"], "adapter-template")
                self.assertEqual(merged["base_only"], 1)
                self.assertEqual(merged["nested"]["base"], "keep")
                self.assertEqual(merged["nested"]["adapter"], "win")
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_ignores_unused_symlinked_extra_files(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            base_target = root / "extra-real.txt"
            base_target.write_text("base-extra", encoding="utf-8")
            (root / "extra-link.txt").symlink_to(base_target)
            adapter_target = adapter / "adapter-real.txt"
            adapter_target.write_text("adapter-extra", encoding="utf-8")
            (adapter / "adapter-link.txt").symlink_to(adapter_target)

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=root, lora_dir=adapter),
                trust_remote_code=False,
            )
            overlay_path = runtime_files.metadata_dir
            try:
                self.assertFalse((overlay_path / "extra-link.txt").exists())
                self.assertFalse((overlay_path / "adapter-link.txt").exists())
                self.assertTrue((overlay_path / "qmodel.tensors").is_file())
                self.assertTrue((overlay_path / "rope.cache").is_file())
                self.assertTrue((overlay_path / "qmodel.lora").is_file())
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_rejects_symlinked_base_config_when_adapter_enabled(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            real_config = root / "config-real.json"
            real_config.write_text((root / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "config.json").unlink()
            (root / "config.json").symlink_to(real_config)

            with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                prepare_runtime_files(
                    InitConfig(model_dir=root, lora_dir=adapter),
                    trust_remote_code=False,
                )

    def test_prepare_runtime_files_rejects_symlinked_runtime_used_tokenizer_file(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            target = root / "tokenizer-real.json"
            target.write_text((root / "tokenizer.json").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "tokenizer.json").unlink()
            (root / "tokenizer.json").symlink_to(target)

            with self.assertRaisesRegex(ModelValidationError, "must not use symlinks"):
                prepare_runtime_files(
                    InitConfig(model_dir=root, lora_dir=adapter),
                    trust_remote_code=False,
                )

    def test_prepare_runtime_files_fails_fast_when_runtime_artifacts_cannot_be_hardlinked(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)

            with patch("trillim.components.llm._model_dir.os.link", side_effect=OSError(errno.EXDEV, "cross-device")):
                with self.assertRaisesRegex(ModelValidationError, "across filesystems"):
                    prepare_runtime_files(
                        InitConfig(model_dir=root, lora_dir=adapter),
                        trust_remote_code=False,
                    )

    def test_prepare_runtime_files_rejects_multi_filesystem_lora_overlays(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)

            def fake_device(path: Path) -> int:
                return 1 if path in (root, adapter) else 2

            with patch("trillim.components.llm._model_dir._filesystem_device", side_effect=fake_device):
                with self.assertRaisesRegex(ModelValidationError, "multi-filesystem"):
                    prepare_runtime_files(
                        InitConfig(model_dir=root, lora_dir=adapter),
                        trust_remote_code=False,
                    )

    def test_prepare_runtime_files_copies_remote_code_with_adapter_override(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenization_adapter.py").write_text(
                "from . import helpers\nHELPER = helpers.HELPER\nSOURCE = 'base'\n",
                encoding="utf-8",
            )
            (root / "helpers.py").write_text("HELPER = 'base-helper'\n", encoding="utf-8")
            (adapter / "tokenization_adapter.py").write_text(
                "from . import helpers\nHELPER = helpers.HELPER\nSOURCE = 'adapter'\n",
                encoding="utf-8",
            )
            (adapter / "extra_module.py").write_text("EXTRA = 1\n", encoding="utf-8")

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=root, lora_dir=adapter),
                trust_remote_code=True,
            )
            overlay_path = runtime_files.metadata_dir
            try:
                self.assertEqual(
                    (overlay_path / "tokenization_adapter.py").read_text(encoding="utf-8"),
                    "from . import helpers\nHELPER = helpers.HELPER\nSOURCE = 'adapter'\n",
                )
                self.assertEqual(
                    (overlay_path / "helpers.py").read_text(encoding="utf-8"),
                    "HELPER = 'base-helper'\n",
                )
                self.assertFalse((overlay_path / "extra_module.py").exists())
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_collects_eos_tokens_from_added_tokens_overlay(self):
        with model_dir(tokenizer_payload={"model": "base"}) as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "added_tokens.json").write_text(
                json.dumps({"<|eot_id|>": 11}),
                encoding="utf-8",
            )
            (adapter / "added_tokens.json").write_text(
                json.dumps({"</s>": 5}),
                encoding="utf-8",
            )

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=root, lora_dir=adapter),
                trust_remote_code=False,
            )
            overlay_path = runtime_files.metadata_dir
            try:
                self.assertEqual(
                    json.loads((overlay_path / "added_tokens.json").read_text(encoding="utf-8")),
                    {"<|eot_id|>": 11, "</s>": 5},
                )

                info = validate_model_dir(root, metadata_dir=overlay_path)

                self.assertEqual(info.eos_tokens[0], 2)
                self.assertCountEqual(info.eos_tokens, (2, 5, 11))
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_rejects_external_remote_code_references(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["repo/name--tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ModelValidationError, "External remote-code repositories"):
                prepare_runtime_files(
                    InitConfig(model_dir=root, lora_dir=adapter),
                    trust_remote_code=True,
                )

    def test_prepare_runtime_files_rejects_package_scoped_auto_map_entry_points(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["pkg.tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ModelValidationError, "currently unsupported"):
                prepare_runtime_files(
                    InitConfig(model_dir=root, lora_dir=adapter),
                    trust_remote_code=True,
                )

    def test_prepare_runtime_files_rejects_parent_relative_remote_code_imports(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenization_adapter.py").write_text(
                "from ..shared import HELPER\nSOURCE = 'base'\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ModelValidationError, "currently unsupported"):
                prepare_runtime_files(
                    InitConfig(model_dir=root, lora_dir=adapter),
                    trust_remote_code=True,
                )

    def test_prepare_runtime_files_rejects_package_scoped_remote_code_imports(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenization_adapter.py").write_text(
                "from .subpackage import HELPER\nSOURCE = 'base'\n",
                encoding="utf-8",
            )
            subpackage = root / "subpackage"
            subpackage.mkdir()
            (subpackage / "__init__.py").write_text("HELPER = 'subpackage'\n", encoding="utf-8")

            with self.assertRaisesRegex(ModelValidationError, "currently unsupported"):
                prepare_runtime_files(
                    InitConfig(model_dir=root, lora_dir=adapter),
                    trust_remote_code=True,
                )

    def test_prepare_runtime_files_rejects_remote_code_file_budget_overflow(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenization_adapter.py").write_text(
                "from .helpers import HELPER\nSOURCE = 'base'\n",
                encoding="utf-8",
            )
            (root / "helpers.py").write_text("HELPER = 'base-helper'\n", encoding="utf-8")

            with patch("trillim.components.llm._model_dir._MAX_REMOTE_CODE_FILES", 1):
                with self.assertRaisesRegex(ModelValidationError, "file budget"):
                    prepare_runtime_files(
                        InitConfig(model_dir=root, lora_dir=adapter),
                        trust_remote_code=True,
                    )

    def test_prepare_runtime_files_rejects_remote_code_depth_overflow(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenization_adapter.py").write_text(
                "from .helpers import HELPER\nSOURCE = 'base'\n",
                encoding="utf-8",
            )
            (root / "helpers.py").write_text("HELPER = 'base-helper'\n", encoding="utf-8")

            with patch("trillim.components.llm._model_dir._MAX_REMOTE_CODE_DEPTH", 0):
                with self.assertRaisesRegex(ModelValidationError, "supported depth"):
                    prepare_runtime_files(
                        InitConfig(model_dir=root, lora_dir=adapter),
                        trust_remote_code=True,
                    )

    def test_prepare_runtime_files_rejects_remote_code_byte_budget_overflow(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoTokenizer": ["tokenization_adapter.AdapterTokenizer", None]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenization_adapter.py").write_text("SOURCE = 'base'\n", encoding="utf-8")

            with patch("trillim.components.llm._model_dir._MAX_REMOTE_CODE_BYTES", 1):
                with self.assertRaisesRegex(ModelValidationError, "byte budget"):
                    prepare_runtime_files(
                        InitConfig(model_dir=root, lora_dir=adapter),
                        trust_remote_code=True,
                    )
