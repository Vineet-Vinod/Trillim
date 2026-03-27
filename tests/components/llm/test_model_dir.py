"""Tests for model-directory validation."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim._bundle_metadata import CURRENT_FORMAT_VERSION
from trillim.components.llm._config import ActivationType, ArchitectureType, InitConfig
from trillim.components.llm._model_dir import (
    _ARCH_REGISTRY,
    _OverlayMetadata,
    _collect_added_tokens,
    _collect_eos_tokens,
    _collect_remote_code_class_refs,
    _collect_remote_code_files,
    _extract_auto_map_value,
    _extract_auto_map_refs,
    _extract_dimensions,
    _filesystem_device,
    _load_json,
    _load_optional_json,
    _load_optional_json_with_message,
    _load_required_json_strict,
    _materialize_file,
    _materialize_required_file,
    _merge_json_payloads,
    _restore_base_auto_map_entry,
    _restore_base_tokenizer_loader_fields,
    _module_name_to_relative_path,
    _parse_remote_code_module_path,
    _relative_import_module_names,
    _require_positive_int,
    _resolve_activation,
    _resolve_directory,
    _resolve_rope_theta,
    prepare_runtime_files,
    validate_lora_dir,
    validate_model_dir,
)
from trillim.components.llm._tokenizer import load_tokenizer
from trillim.errors import ModelValidationError
from tests.components.llm.support import model_dir, write_adapter_bundle
from tests.components.llm.support import write_model_bundle


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

    def test_validate_model_dir_requires_current_bundle_metadata(self):
        with model_dir() as root:
            (root / "trillim_config.json").unlink()
            with self.assertRaisesRegex(ModelValidationError, "missing or unsupported"):
                validate_model_dir(root)

        with model_dir() as root:
            (root / "trillim_config.json").write_text(
                json.dumps({"format_version": CURRENT_FORMAT_VERSION - 1}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelValidationError, "missing or unsupported"):
                validate_model_dir(root)

    def test_validate_model_dir_rejects_non_object_bundle_metadata(self):
        with model_dir() as root:
            (root / "trillim_config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "missing or unsupported"):
                validate_model_dir(root)

    def test_validate_model_dir_rejects_non_object_config_json(self):
        with model_dir() as root:
            (root / "config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "config.json must be a JSON object"):
                validate_model_dir(root)

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

    def test_validate_model_dir_fails_closed_on_malformed_added_tokens_entries(self):
        for filename in ("tokenizer.json", "added_tokens.json"):
            with self.subTest(filename=filename):
                with model_dir() as root:
                    (root / filename).write_text(
                        json.dumps({"added_tokens": ["oops"]}),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ModelValidationError, "added_tokens"):
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
                json.dumps({"format_version": CURRENT_FORMAT_VERSION - 1}),
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

    def test_validate_lora_dir_wraps_invalid_base_model_config_as_validation_error(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)

            for invalid_payload in (None, "{"):
                with self.subTest(invalid_payload=invalid_payload):
                    if invalid_payload is None:
                        (root / "config.json").unlink()
                    else:
                        (root / "config.json").write_text(invalid_payload, encoding="utf-8")

                    with self.assertRaisesRegex(
                        ModelValidationError,
                        "Could not validate adapter compatibility",
                    ):
                        validate_lora_dir(adapter, model_dir=root)

                    write_model = {
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 128,
                        "intermediate_size": 256,
                        "num_attention_heads": 4,
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 4,
                        "vocab_size": 256,
                        "max_position_embeddings": 512,
                        "hidden_act": "silu",
                        "eos_token_id": 2,
                    }
                    (root / "config.json").write_text(
                        json.dumps(write_model),
                        encoding="utf-8",
                    )

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

    def test_prepare_runtime_files_preserves_base_loader_fields_without_explicit_adapter_override(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import PreTrainedTokenizerFast

            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            backend = Tokenizer(
                WordLevel({"<unk>": 0, "<pad>": 1, "hello": 2}, unk_token="<unk>")
            )
            backend.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="<unk>",
                pad_token="<pad>",
            ).save_pretrained(str(root))
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 128,
                        "intermediate_size": 256,
                        "num_attention_heads": 4,
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 4,
                        "vocab_size": 256,
                        "max_position_embeddings": 512,
                        "hidden_act": "silu",
                        "eos_token_id": 2,
                        "tokenizer_class": "BaseConfigTokenizer",
                        "auto_map": {"AutoConfig": "config_mod.BaseConfig"},
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": 5,
                        "tokenizer_class": "TokenizersBackend",
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "PreTrainedTokenizerFast",
                        "chat_template": "base-template",
                        "unk_token": "<unk>",
                        "pad_token": "<pad>",
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "TokenizersBackend",
                        "chat_template": "adapter-template",
                        "pad_token": "<pad>",
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
                merged_config = json.loads((overlay_path / "config.json").read_text(encoding="utf-8"))
                merged_tokenizer_config = json.loads(
                    (overlay_path / "tokenizer_config.json").read_text(encoding="utf-8")
                )

                self.assertEqual(merged_config["eos_token_id"], 5)
                self.assertEqual(merged_config["tokenizer_class"], "BaseConfigTokenizer")
                self.assertEqual(
                    merged_config["auto_map"],
                    {"AutoConfig": "config_mod.BaseConfig"},
                )
                self.assertEqual(
                    merged_tokenizer_config["tokenizer_class"],
                    "PreTrainedTokenizerFast",
                )
                self.assertEqual(merged_tokenizer_config["chat_template"], "adapter-template")
                self.assertEqual(merged_tokenizer_config["pad_token"], "<pad>")

                tokenizer = load_tokenizer(overlay_path, trust_remote_code=False)

                self.assertTrue(callable(getattr(tokenizer, "encode", None)))
                self.assertTrue(callable(getattr(tokenizer, "decode", None)))
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_drops_implicit_adapter_loader_fields_when_base_has_none(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 128,
                        "intermediate_size": 256,
                        "num_attention_heads": 4,
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 4,
                        "vocab_size": 256,
                        "max_position_embeddings": 512,
                        "hidden_act": "silu",
                        "eos_token_id": 2,
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": 5,
                        "tokenizer_class": "TokenizersBackend",
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "base-template"}),
                encoding="utf-8",
            )
            (adapter / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "TokenizersBackend",
                        "chat_template": "adapter-template",
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
                merged_config = json.loads((overlay_path / "config.json").read_text(encoding="utf-8"))
                merged_tokenizer_config = json.loads(
                    (overlay_path / "tokenizer_config.json").read_text(encoding="utf-8")
                )

                self.assertEqual(merged_config["eos_token_id"], 5)
                self.assertNotIn("tokenizer_class", merged_config)
                self.assertEqual(merged_tokenizer_config["chat_template"], "adapter-template")
                self.assertNotIn("tokenizer_class", merged_tokenizer_config)
            finally:
                runtime_files.cleanup()

    def test_prepare_runtime_files_preserves_explicit_adapter_auto_tokenizer_override(self):
        with model_dir() as root, tempfile.TemporaryDirectory() as temp_dir:
            adapter = Path(temp_dir) / "adapter"
            write_adapter_bundle(adapter, model_root=root)
            (root / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
                encoding="utf-8",
            )
            (adapter / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "AdapterTokenizer",
                        "auto_map": ["tokenization_adapter.AdapterTokenizer", None],
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

                self.assertEqual(merged["tokenizer_class"], "AdapterTokenizer")
                self.assertEqual(
                    merged["auto_map"],
                    ["tokenization_adapter.AdapterTokenizer", None],
                )
            finally:
                runtime_files.cleanup()

    def test_restore_base_tokenizer_loader_fields_drops_unowned_loader_state(self):
        merged = {
            "tokenizer_class": "AdapterTokenizer",
            "auto_map": ["tokenization_adapter.AdapterTokenizer", None],
        }

        _restore_base_tokenizer_loader_fields(merged, {"tokenizer_class": ""})

        self.assertNotIn("tokenizer_class", merged)
        self.assertNotIn("auto_map", merged)

        merged = {
            "tokenizer_class": "AdapterTokenizer",
            "auto_map": ["tokenization_adapter.AdapterTokenizer", None],
        }
        _restore_base_tokenizer_loader_fields(merged, None)
        self.assertNotIn("tokenizer_class", merged)
        self.assertNotIn("auto_map", merged)

    def test_restore_base_auto_map_entry_covers_list_and_dict_cleanup_paths(self):
        merged = {"auto_map": {"AutoTokenizer": "adapter.Tokenizer"}}
        _restore_base_auto_map_entry(
            merged,
            {"auto_map": ["base.Tokenizer", None]},
            key="AutoTokenizer",
        )
        self.assertEqual(merged["auto_map"], ["base.Tokenizer", None])

        merged = {"auto_map": {"AutoTokenizer": "adapter.Tokenizer", "Other": "keep"}}
        _restore_base_auto_map_entry(
            merged,
            {"auto_map": {"AutoConfig": "config_mod.Config"}},
            key="AutoTokenizer",
        )
        self.assertEqual(merged["auto_map"], {"Other": "keep"})

        merged = {"auto_map": {"AutoTokenizer": "adapter.Tokenizer"}}
        _restore_base_auto_map_entry(
            merged,
            {"auto_map": {"AutoConfig": "config_mod.Config"}},
            key="AutoTokenizer",
        )
        self.assertNotIn("auto_map", merged)

        merged = {"auto_map": {"Other": "keep"}}
        _restore_base_auto_map_entry(merged, None, key="AutoTokenizer")
        self.assertEqual(merged["auto_map"], {"Other": "keep"})

        merged = {"auto_map": {"AutoTokenizer": "adapter.Tokenizer"}}
        _restore_base_auto_map_entry(merged, None, key="AutoTokenizer")
        self.assertNotIn("auto_map", merged)

        merged = {"auto_map": {"AutoTokenizer": "adapter.Tokenizer", "Other": "keep"}}
        _restore_base_auto_map_entry(merged, None, key="AutoTokenizer")
        self.assertEqual(merged["auto_map"], {"Other": "keep"})

        merged = {}
        _restore_base_auto_map_entry(
            merged,
            {
                "auto_map": {
                    "AutoTokenizer": ["base.Tokenizer", None],
                    "AutoConfig": "config_mod.Config",
                }
            },
            key="AutoTokenizer",
        )
        self.assertEqual(
            merged["auto_map"],
            {
                "AutoTokenizer": ["base.Tokenizer", None],
                "AutoConfig": "config_mod.Config",
            },
        )

    def test_extract_auto_map_value_handles_invalid_and_sequence_values(self):
        self.assertEqual(
            _extract_auto_map_value(None, key="AutoTokenizer"),
            (None, False, False),
        )
        self.assertEqual(
            _extract_auto_map_value({"auto_map": ["", None]}, key="AutoTokenizer"),
            (None, False, False),
        )
        self.assertEqual(
            _extract_auto_map_value(
                {"auto_map": {"AutoTokenizer": ["", None]}},
                key="AutoTokenizer",
            ),
            (None, False, False),
        )
        self.assertEqual(
            _extract_auto_map_value(
                {"auto_map": {"AutoTokenizer": ("base.Tokenizer", None)}},
                key="AutoTokenizer",
            ),
            (["base.Tokenizer", None], True, False),
        )
        self.assertEqual(
            _extract_auto_map_value(
                {"auto_map": {"AutoTokenizer": ["base.Tokenizer", None]}},
                key="AutoTokenizer",
            ),
            (["base.Tokenizer", None], True, False),
        )
        self.assertEqual(
            _extract_auto_map_value(
                {"auto_map": {"AutoTokenizer": "base.Tokenizer"}},
                key="AutoTokenizer",
            ),
            ("base.Tokenizer", True, False),
        )

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

    def test_model_dir_private_json_helpers_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            broken = root / "broken.json"
            broken.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ModelValidationError, "Could not read JSON"):
                _load_json(broken)

            self.assertIsNone(_load_optional_json(broken))
            self.assertIsNone(
                _load_optional_json_with_message(
                    broken,
                    symlink_message="Model bundle must not use symlinks",
                )
            )

            with self.assertRaisesRegex(ModelValidationError, "missing.json not found"):
                _load_required_json_strict(
                    root / "missing.json",
                    symlink_message="Model bundle must not use symlinks",
                )

    def test_model_dir_numeric_and_activation_helpers_cover_error_paths(self):
        arch_info = _ARCH_REGISTRY["llamaforcausallm"]
        self.assertEqual(_resolve_activation({}, arch_info), ActivationType.SILU)

        with self.assertRaisesRegex(ModelValidationError, "Unsupported activation function"):
            _resolve_activation({"hidden_act": "bogus"}, arch_info)

        with self.assertRaisesRegex(ModelValidationError, "rope_theta must be numeric"):
            _resolve_rope_theta({"rope_theta": "bogus"})

        with self.assertRaisesRegex(ModelValidationError, "head_dim must be a positive integer"):
            _extract_dimensions(
                {
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 4,
                    "vocab_size": 256,
                    "max_position_embeddings": 512,
                    "head_dim": 0,
                }
            )

        for value in (True, "bad", 0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ModelValidationError, "hidden_size must be a positive integer"):
                    _require_positive_int(value, "hidden_size")

    def test_model_dir_eos_and_added_token_helpers_cover_edge_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_dir = Path(temp_dir)
            self.assertEqual(
                _collect_eos_tokens(
                    {"eos_token_id": [1, 2]},
                    ArchitectureType.LLAMA,
                    metadata_dir,
                ),
                [1, 2],
            )

            with self.assertRaisesRegex(ModelValidationError, "No EOS tokens"):
                _collect_eos_tokens(
                    {"eos_token_id": []},
                    ArchitectureType.LLAMA,
                    metadata_dir,
                )

        with self.assertRaisesRegex(ModelValidationError, "added_tokens metadata is malformed"):
            _collect_added_tokens([{"content": "</s>", "id": 2}])

        with self.assertRaisesRegex(ModelValidationError, "added_tokens metadata is malformed"):
            _collect_added_tokens({"added_tokens": "oops"})

        self.assertEqual(
            _collect_added_tokens({"added_tokens": [{"content": "other"}]}),
            [],
        )
        self.assertEqual(_collect_added_tokens("oops"), [])

    def test_model_dir_directory_and_materialization_helpers_cover_missing_and_oserror_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_path = root / "file.txt"
            file_path.write_text("data", encoding="utf-8")

            with self.assertRaisesRegex(ModelValidationError, "is not a directory"):
                _resolve_directory(
                    file_path,
                    label="Model directory",
                    symlink_message="Model directory must not use symlinks",
                )

            overlay = root / "overlay"
            source_dir = root / "source"
            source_dir.mkdir()
            with self.assertRaisesRegex(ModelValidationError, "missing.txt not found"):
                _materialize_required_file(
                    overlay,
                    source_dir=source_dir,
                    relative_path=Path("missing.txt"),
                    symlink_message="Model bundle must not use symlinks",
                    mode="copy",
                )

            source_path = source_dir / "source.txt"
            source_path.write_text("fresh", encoding="utf-8")
            destination = overlay / "source.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("stale", encoding="utf-8")
            _materialize_file(destination, source_path, mode="copy")
            self.assertEqual(destination.read_text(encoding="utf-8"), "fresh")

            with patch(
                "trillim.components.llm._model_dir.os.link",
                side_effect=OSError(errno.EPERM, "denied"),
            ):
                with self.assertRaisesRegex(ModelValidationError, "Could not hardlink runtime artifact"):
                    _materialize_file(overlay / "linked.bin", source_path, mode="hardlink")

        with patch.object(Path, "stat", side_effect=OSError("boom")):
            with self.assertRaisesRegex(ModelValidationError, "Could not inspect filesystem"):
                _filesystem_device(Path("/tmp"))

    def test_model_dir_remote_code_helpers_cover_duplicates_and_invalid_inputs(self):
        self.assertEqual(_extract_auto_map_refs(None, key="AutoTokenizer"), [])
        self.assertEqual(
            _extract_auto_map_refs(
                {"auto_map": ["tokenization.AdapterTokenizer", None]},
                key="AutoTokenizer",
            ),
            ["tokenization.AdapterTokenizer"],
        )

        with self.assertRaisesRegex(ModelValidationError, "currently unsupported"):
            _parse_remote_code_module_path("BrokenRef")

        with self.assertRaisesRegex(ModelValidationError, "currently unsupported"):
            _module_name_to_relative_path("")

        metadata = _OverlayMetadata(
            config={"auto_map": {"AutoConfig": "tokenization.AdapterTokenizer"}},
            added_tokens=None,
            generation_config=None,
            special_tokens_map=None,
            tokenizer_config={
                "auto_map": {
                    "AutoTokenizer": ["tokenization.AdapterTokenizer", None]
                }
            },
        )
        self.assertEqual(
            _collect_remote_code_class_refs(metadata),
            ["tokenization.AdapterTokenizer"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_root = root / "model"
            adapter_root = root / "adapter"
            model_root.mkdir()
            adapter_root.mkdir()
            (model_root / "tokenization.py").write_text(
                "from os import path\nfrom . import helper\nfrom . import helper\nfrom . import *\n",
                encoding="utf-8",
            )
            (model_root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

            self.assertEqual(
                _relative_import_module_names(model_root / "tokenization.py"),
                ["helper"],
            )
            self.assertEqual(
                _collect_remote_code_files(model_root, adapter_root, metadata),
                [Path("tokenization.py"), Path("helper.py")],
            )

            (model_root / "fanout.py").write_text(
                "from . import helper\nfrom . import branch\n",
                encoding="utf-8",
            )
            (model_root / "branch.py").write_text("from . import helper\n", encoding="utf-8")
            duplicate_queue_metadata = _OverlayMetadata(
                config=None,
                added_tokens=None,
                generation_config=None,
                special_tokens_map=None,
                tokenizer_config={
                    "auto_map": {
                        "AutoTokenizer": ["fanout.AdapterTokenizer", None]
                    }
                },
            )
            self.assertEqual(
                _collect_remote_code_files(model_root, adapter_root, duplicate_queue_metadata),
                [Path("fanout.py"), Path("helper.py"), Path("branch.py")],
            )

            (model_root / "cycle_a.py").write_text(
                "from . import cycle_b\n",
                encoding="utf-8",
            )
            (model_root / "cycle_b.py").write_text(
                "from . import cycle_a\n",
                encoding="utf-8",
            )
            cycle_metadata = _OverlayMetadata(
                config=None,
                added_tokens=None,
                generation_config=None,
                special_tokens_map=None,
                tokenizer_config={
                    "auto_map": {
                        "AutoTokenizer": ["cycle_a.AdapterTokenizer", None]
                    }
                },
            )
            self.assertEqual(
                _collect_remote_code_files(model_root, adapter_root, cycle_metadata),
                [Path("cycle_a.py"), Path("cycle_b.py")],
            )

            with self.assertRaisesRegex(ModelValidationError, "Could not read Python module"):
                _relative_import_module_names(model_root / "missing.py")

            broken = model_root / "broken.py"
            broken.write_text("def broken(:\n", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "Could not parse Python module"):
                _relative_import_module_names(broken)

            package_scoped = model_root / "package_scoped.py"
            package_scoped.write_text("from .pkg.sub import helper\n", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "Package-scoped relative imports"):
                _relative_import_module_names(package_scoped)

            missing_metadata = _OverlayMetadata(
                config=None,
                added_tokens=None,
                generation_config=None,
                special_tokens_map=None,
                tokenizer_config={
                    "auto_map": {
                        "AutoTokenizer": ["missing.AdapterTokenizer", None]
                    }
                },
            )
            with self.assertRaisesRegex(ModelValidationError, "Remote-code module not found"):
                _collect_remote_code_files(model_root, adapter_root, missing_metadata)

    def test_model_dir_merge_json_payloads_prefers_override_for_non_dict_payloads(self):
        self.assertEqual(_merge_json_payloads(["base"], ["override"]), ["override"])

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
