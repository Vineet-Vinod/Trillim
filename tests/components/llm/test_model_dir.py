from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trillim import _model_store
from trillim.components.llm._config import ActivationType, ArchitectureType, InitConfig
from trillim.components.llm._model_dir import (
    _OverlayMetadata,
    _collect_remote_code_files,
    _module_name_to_relative_path,
    _parse_remote_code_module_path,
    _relative_import_module_names,
    _resolve_relative_import_module_path,
    prepare_runtime_files,
    validate_lora_dir,
    validate_model_dir,
)
from trillim.errors import ModelValidationError

from tests.support import write_llm_bundle, write_lora_bundle


BITNET_MODEL_DIR = _model_store.store_path_for_id("Trillim/BitNet-TRNQ")
BITNET_SEARCH_ADAPTER_DIR = _model_store.store_path_for_id(
    "Trillim/BitNet-Search-LoRA-TRNQ"
)
BONSAI_TERNARY_MODEL_DIR = _model_store.store_path_for_id("Trillim/Bonsai-1.7BT-TRNQ")


class ModelDirTests(unittest.TestCase):
    def test_validate_model_dir_extracts_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = write_llm_bundle(Path(temp_dir) / "model")

            config = validate_model_dir(model_dir)

        self.assertEqual(config.name, "model")
        self.assertEqual(config.arch_type, ArchitectureType.LLAMA)
        self.assertEqual(config.activation, ActivationType.SILU)
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.intermediate_dim, 384)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_kv_heads, 2)
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(config.rope_theta, 12000.0)
        self.assertEqual(config.eos_tokens, (2, 3, 151645))
        self.assertTrue(config.tie_word_embeddings)

    def test_validate_model_dir_supports_text_config_and_rejects_bad_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = write_llm_bundle(
                root / "text-config",
                text_config={
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_attention_heads": 4,
                    "num_hidden_layers": 2,
                    "vocab_size": 100,
                    "hidden_act": "swish",
                    "eos_token_id": 2,
                },
            )
            config = validate_model_dir(model_dir)
            self.assertEqual(config.activation, ActivationType.SILU)

            bad_activation = write_llm_bundle(
                root / "bad-activation",
                hidden_act="unsupported",
            )
            with self.assertRaisesRegex(ModelValidationError, "Unsupported activation"):
                validate_model_dir(bad_activation)

            bad_rope = write_llm_bundle(
                root / "bad-rope",
                config_overrides={"rope_theta": "not numeric"},
            )
            with self.assertRaisesRegex(ModelValidationError, "rope_theta"):
                validate_model_dir(bad_rope)

    def test_validate_model_dir_rejects_missing_artifact_and_unknown_architecture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = write_llm_bundle(Path(temp_dir) / "model")
            (model_dir / "rope.cache").unlink()
            with self.assertRaisesRegex(ModelValidationError, "rope.cache"):
                validate_model_dir(model_dir)

            unsupported = write_llm_bundle(
                Path(temp_dir) / "unsupported",
                architecture="UnknownForCausalLM",
            )
            with self.assertRaisesRegex(ModelValidationError, "Unsupported"):
                validate_model_dir(unsupported)

            (unsupported / "trillim_config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "metadata"):
                validate_model_dir(unsupported)

    def test_validate_model_dir_rejects_symlinked_required_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = write_llm_bundle(root / "model")
            target = root / "config-target.json"
            target.write_text((model_dir / "config.json").read_text(), encoding="utf-8")
            (model_dir / "config.json").unlink()
            (model_dir / "config.json").symlink_to(target)

            with self.assertRaisesRegex(ModelValidationError, "symlinks"):
                validate_model_dir(model_dir)

    def test_validate_model_dir_rejects_bad_paths_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaisesRegex(ModelValidationError, "does not exist"):
                validate_model_dir(root / "missing")

            file_path = root / "not-a-dir"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "not a directory"):
                validate_model_dir(file_path)

            model_dir = write_llm_bundle(root / "model")
            (model_dir / "config.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "Could not read JSON"):
                validate_model_dir(model_dir)

            (model_dir / "config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "JSON object"):
                validate_model_dir(model_dir)

    def test_validate_model_dir_rejects_bad_dimensions_and_token_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            bool_dimension = write_llm_bundle(
                root / "bool-dimension",
                config_overrides={"hidden_size": True},
            )
            with self.assertRaisesRegex(ModelValidationError, "hidden_size"):
                validate_model_dir(bool_dimension)

            bad_head_dim = write_llm_bundle(
                root / "bad-head-dim",
                config_overrides={"head_dim": 0},
            )
            with self.assertRaisesRegex(ModelValidationError, "head_dim"):
                validate_model_dir(bad_head_dim)

            bad_added_tokens = write_llm_bundle(
                root / "bad-added-tokens",
                tokenizer_payload={"added_tokens": "bad"},
            )
            with self.assertRaisesRegex(ModelValidationError, "added_tokens"):
                validate_model_dir(bad_added_tokens)

    def test_validate_lora_dir_checks_artifacts_and_base_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = write_llm_bundle(root / "model")
            adapter_dir = write_lora_bundle(root / "adapter", model_dir=model_dir)

            self.assertEqual(validate_lora_dir(adapter_dir, model_dir=model_dir), adapter_dir)

            payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            payload["hidden_size"] = 512
            (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "mismatch"):
                validate_lora_dir(adapter_dir, model_dir=model_dir)

    def test_validate_lora_dir_rejects_missing_metadata_and_bad_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = write_llm_bundle(root / "model")

            with self.assertRaisesRegex(ModelValidationError, "does not exist"):
                validate_lora_dir(root / "missing")

            adapter_dir = write_lora_bundle(root / "adapter", model_dir=model_dir)
            (adapter_dir / "qmodel.lora").unlink()
            with self.assertRaisesRegex(ModelValidationError, "qmodel.lora"):
                validate_lora_dir(adapter_dir)

            bad_metadata = write_lora_bundle(root / "bad-metadata", model_dir=model_dir)
            (bad_metadata / "trillim_config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "JSON object"):
                validate_lora_dir(bad_metadata)

            unsupported = write_lora_bundle(root / "unsupported", model_dir=model_dir)
            payload = json.loads(
                (unsupported / "trillim_config.json").read_text(encoding="utf-8")
            )
            payload["format_version"] = 0
            (unsupported / "trillim_config.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelValidationError, "compatibility metadata"):
                validate_lora_dir(unsupported)

    def test_prepare_runtime_files_builds_and_cleans_lora_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = write_llm_bundle(root / "model")
            adapter_dir = write_lora_bundle(root / "adapter", model_dir=model_dir)
            (adapter_dir / "tokenizer_config.json").write_text(
                json.dumps({"extra": "adapter"}),
                encoding="utf-8",
            )

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=model_dir, lora_dir=adapter_dir),
                trust_remote_code=False,
            )
            overlay = runtime_files.metadata_dir

            self.assertEqual(runtime_files.model_dir, model_dir.resolve())
            self.assertEqual(runtime_files.adapter_dir, adapter_dir.resolve())
            self.assertTrue((overlay / "qmodel.tensors").is_file())
            self.assertTrue((overlay / "qmodel.lora").is_file())
            self.assertTrue((overlay / "tokenizer_config.json").is_file())
            runtime_files.cleanup()
            self.assertFalse(overlay.exists())

    def test_prepare_runtime_files_without_lora_uses_model_dir_directly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = write_llm_bundle(Path(temp_dir) / "model")

            runtime_files = prepare_runtime_files(
                InitConfig(model_dir=model_dir),
                trust_remote_code=False,
            )

            self.assertEqual(runtime_files.model_dir, model_dir.resolve())
            self.assertEqual(runtime_files.metadata_dir, model_dir.resolve())
            self.assertIsNone(runtime_files.adapter_dir)
            runtime_files.cleanup()

    @unittest.skipUnless(
        BITNET_MODEL_DIR.is_dir()
        and BITNET_SEARCH_ADAPTER_DIR.is_dir()
        and BONSAI_TERNARY_MODEL_DIR.is_dir(),
        "real BitNet, search adapter, and Bonsai ternary bundles must be installed",
    )
    def test_real_installed_model_and_adapter_metadata_paths(self):
        bitnet = validate_model_dir(BITNET_MODEL_DIR)
        bonsai_ternary = validate_model_dir(BONSAI_TERNARY_MODEL_DIR)
        adapter = validate_lora_dir(BITNET_SEARCH_ADAPTER_DIR, model_dir=BITNET_MODEL_DIR)

        self.assertEqual(bitnet.arch_type, ArchitectureType.BITNET)
        self.assertTrue(bitnet.has_attn_sub_norm)
        self.assertEqual(bonsai_ternary.arch_type, ArchitectureType.BONSAI_TERNARY)
        self.assertEqual(adapter, BITNET_SEARCH_ADAPTER_DIR)

    @unittest.skipUnless(
        BITNET_MODEL_DIR.is_dir() and BITNET_SEARCH_ADAPTER_DIR.is_dir(),
        "real BitNet and search adapter bundles must be installed",
    )
    def test_real_search_adapter_overlay_uses_adapter_tokenizer_metadata(self):
        runtime_files = prepare_runtime_files(
            InitConfig(model_dir=BITNET_MODEL_DIR, lora_dir=BITNET_SEARCH_ADAPTER_DIR),
            trust_remote_code=False,
        )
        try:
            overlay = runtime_files.metadata_dir
            self.assertEqual(runtime_files.model_dir, BITNET_MODEL_DIR)
            self.assertEqual(runtime_files.adapter_dir, BITNET_SEARCH_ADAPTER_DIR)
            self.assertTrue((overlay / "qmodel.tensors").is_file())
            self.assertTrue((overlay / "qmodel.lora").is_file())
            self.assertTrue((overlay / "tokenizer.json").is_file())
            self.assertTrue((overlay / "tokenizer_config.json").is_file())
            self.assertEqual(validate_model_dir(runtime_files.model_dir, metadata_dir=overlay).name, "BitNet-TRNQ")
        finally:
            runtime_files.cleanup()

    def test_remote_code_helpers_collect_local_modules_and_reject_bad_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            (model_dir / "tokenization_local.py").write_text(
                "from .helper import Helper\nfrom . import sidecar\n",
                encoding="utf-8",
            )
            (model_dir / "helper.py").write_text("class Helper: pass\n", encoding="utf-8")
            (model_dir / "sidecar.py").write_text("VALUE = 1\n", encoding="utf-8")
            metadata = _OverlayMetadata(
                config={"auto_map": {"AutoConfig": "tokenization_local.Config"}},
                added_tokens=None,
                generation_config=None,
                special_tokens_map=None,
                tokenizer_config={
                    "auto_map": {
                        "AutoTokenizer": [
                            "tokenization_local.Tokenizer",
                            "tokenization_local.FastTokenizer",
                        ]
                    }
                },
            )

            collected = _collect_remote_code_files(model_dir, adapter_dir, metadata)

            self.assertEqual(
                collected,
                [Path("tokenization_local.py"), Path("helper.py"), Path("sidecar.py")],
            )
            self.assertEqual(_parse_remote_code_module_path("tokenization_local.Tokenizer"), Path("tokenization_local.py"))
            self.assertEqual(_module_name_to_relative_path("helper"), Path("helper.py"))
            self.assertEqual(
                _relative_import_module_names(model_dir / "tokenization_local.py"),
                ["helper", "sidecar"],
            )
            with self.assertRaisesRegex(ModelValidationError, "External"):
                _parse_remote_code_module_path("repo--module.Tokenizer")
            with self.assertRaisesRegex(ModelValidationError, "unsupported"):
                _parse_remote_code_module_path("Tokenizer")
            with self.assertRaisesRegex(ModelValidationError, "Package-scoped"):
                _parse_remote_code_module_path("pkg.module.Tokenizer")
            with self.assertRaisesRegex(ModelValidationError, "unsupported"):
                _module_name_to_relative_path("")

    def test_remote_code_helpers_reject_import_graph_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()

            syntax_error = model_dir / "syntax_error.py"
            syntax_error.write_text("def nope(:\n", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "parse Python module"):
                _relative_import_module_names(syntax_error)

            parent_import = model_dir / "parent_import.py"
            parent_import.write_text("from ..parent import Thing\n", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "Parent relative imports"):
                _relative_import_module_names(parent_import)

            package_import = model_dir / "package_import.py"
            package_import.write_text("from .pkg.module import Thing\n", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "Package-scoped relative imports"):
                _relative_import_module_names(package_import)

            (model_dir / "pkg").mkdir()
            (model_dir / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ModelValidationError, "Package-scoped"):
                _resolve_relative_import_module_path(
                    model_dir,
                    adapter_dir,
                    source_relative_path=Path("tokenization_local.py"),
                    module_name="pkg",
                )
