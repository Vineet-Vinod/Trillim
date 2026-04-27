from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim import _model_store
from trillim.components.llm._config import ArchitectureType
from trillim.quantize import _output
from trillim.quantize._config import load_model_config
from trillim.quantize._output import (
    build_staging_dir,
    copy_adapter_support_files,
    copy_model_support_files,
    mark_staging_complete,
    prepare_output_target,
    publish_staging_dir,
    recover_publish_state,
    write_adapter_metadata,
    write_model_metadata,
    _collect_remote_code_files,
    _parse_remote_code_module_path,
    _quantization_name,
)

from tests.quantize.test_config_manifest import _write_config


class QuantizeOutputTests(unittest.TestCase):
    def test_publish_recover_and_prepare_output_target_use_managed_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            local_root = root / "Local"
            with patch.object(_model_store, "LOCAL_ROOT", local_root):
                with patch.object(_output, "_should_prompt_for_overwrite", return_value=False):
                    target = prepare_output_target(source)
                    self.assertEqual(target, local_root / "source-TRNQ")

                    staging = build_staging_dir(target)
                    (staging / "payload.txt").write_text("new", encoding="utf-8")
                    mark_staging_complete(staging)
                    publish_staging_dir(target)
                    self.assertEqual((target / "payload.txt").read_text(encoding="utf-8"), "new")

                    replacement = build_staging_dir(target)
                    (replacement / "payload.txt").write_text("replacement", encoding="utf-8")
                    mark_staging_complete(replacement)
                    publish_staging_dir(target)
                    self.assertEqual(
                        (target / "payload.txt").read_text(encoding="utf-8"),
                        "replacement",
                    )

                    stale_target = local_root / "stale"
                    stale_staging = local_root / "stale-new"
                    stale_staging.mkdir()
                    recover_publish_state(stale_target)
                    self.assertFalse(stale_staging.exists())

                    deduped = prepare_output_target(source)
                    self.assertEqual(deduped, local_root / "source-TRNQ-2")

    def test_copy_model_support_files_copies_allowlist_and_remote_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            output_dir = root / "out"
            model_dir.mkdir()
            _write_config(model_dir)
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "LocalTokenizer"}),
                encoding="utf-8",
            )
            (model_dir / "tokenization_local.py").write_text(
                "from .helper import Helper\nclass LocalTokenizer: pass\n",
                encoding="utf-8",
            )
            (model_dir / "helper.py").write_text("class Helper: pass\n", encoding="utf-8")
            (model_dir / "ignore.bin").write_bytes(b"ignored")

            copy_model_support_files(model_dir, output_dir)

            tokenizer_config = json.loads(
                (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                tokenizer_config["auto_map"]["AutoTokenizer"],
                ["tokenization_local.LocalTokenizer", None],
            )
            self.assertTrue((output_dir / "tokenization_local.py").is_file())
            self.assertTrue((output_dir / "helper.py").is_file())
            self.assertFalse((output_dir / "ignore.bin").exists())

    def test_copy_adapter_support_files_sanitizes_inherited_tokenizer_loader_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_dir = root / "adapter"
            output_dir = root / "out"
            (adapter_dir / "__pycache__").mkdir(parents=True)
            (adapter_dir / "nested").mkdir()
            (adapter_dir / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "BaseTokenizer",
                        "auto_map": {"Other": "x"},
                    }
                ),
                encoding="utf-8",
            )
            (adapter_dir / "config.json").write_text(
                json.dumps({"auto_map": {"AutoConfig": "adapter.Config"}}),
                encoding="utf-8",
            )
            (adapter_dir / "nested" / "keep.txt").write_text("keep", encoding="utf-8")
            (adapter_dir / "qmodel.lora").write_bytes(b"skip")
            (adapter_dir / "__pycache__" / "skip.pyc").write_bytes(b"skip")

            copy_adapter_support_files(adapter_dir, output_dir)

            tokenizer_config = json.loads(
                (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("tokenizer_class", tokenizer_config)
            self.assertEqual(tokenizer_config["auto_map"], {"Other": "x"})
            self.assertEqual(
                (output_dir / "nested" / "keep.txt").read_text(encoding="utf-8"),
                "keep",
            )
            self.assertFalse((output_dir / "qmodel.lora").exists())
            self.assertFalse((output_dir / "__pycache__").exists())

    def test_write_model_and_adapter_metadata_use_real_config_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_out = root / "model-out"
            adapter_out = root / "adapter-out"
            model_dir.mkdir()
            adapter_dir.mkdir()
            _write_config(model_dir, _name_or_path="source-model")
            config = load_model_config(model_dir)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "base-model"}),
                encoding="utf-8",
            )
            (adapter_dir / "config.json").write_text(
                json.dumps({"auto_map": {"AutoConfig": "adapter.Config"}}),
                encoding="utf-8",
            )
            (adapter_dir / "adapter.py").write_text("class Config: pass\n", encoding="utf-8")

            write_model_metadata(model_out, config=config, model_dir=model_dir)
            write_adapter_metadata(
                adapter_out,
                config=config,
                adapter_dir=adapter_dir,
                model_dir=model_dir,
            )

            model_payload = json.loads(
                (model_out / "trillim_config.json").read_text(encoding="utf-8")
            )
            adapter_payload = json.loads(
                (adapter_out / "trillim_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(model_payload["type"], "model")
            self.assertEqual(model_payload["source_model"], "source-model")
            self.assertEqual(adapter_payload["type"], "lora_adapter")
            self.assertEqual(adapter_payload["source_model"], "base-model")
            self.assertTrue(adapter_payload["remote_code"])

    def test_remote_code_reference_validation_and_quantization_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text(
                json.dumps({"auto_map": {"AutoConfig": "modeling_local.Config"}}),
                encoding="utf-8",
            )
            (model_dir / "modeling_local.py").write_text(
                "from .layers import Layer\nclass Config: pass\n",
                encoding="utf-8",
            )
            (model_dir / "layers.py").write_text("class Layer: pass\n", encoding="utf-8")

            self.assertEqual(
                _collect_remote_code_files(model_dir),
                [Path("modeling_local.py"), Path("layers.py")],
            )

        with self.assertRaisesRegex(ValueError, "External remote-code"):
            _parse_remote_code_module_path("other--repo.module.Class")
        self.assertEqual(_quantization_name(ArchitectureType.BONSAI), "binary")
        self.assertEqual(
            _quantization_name(ArchitectureType.BONSAI_TERNARY),
            "grouped-ternary",
        )
        self.assertEqual(_quantization_name(ArchitectureType.LLAMA), "ternary")
