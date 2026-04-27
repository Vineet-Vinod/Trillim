from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim import _model_store
from trillim.quantize._entrypoint import quantize, _normalize_source_dir

from tests.quantize.test_config_manifest import _write_config, _write_safetensors


class QuantizeEntrypointTests(unittest.TestCase):
    def test_normalize_source_dir_rejects_files_and_model_store_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_file = root / "file"
            source_file.write_text("x", encoding="utf-8")
            models_root = root / "models"
            local_model = models_root / "Local" / "model"
            local_model.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "not a directory"):
                _normalize_source_dir(source_file, label="Model directory")
            with patch.object(_model_store, "MODELS_ROOT", models_root):
                with self.assertRaisesRegex(ValueError, "must not be inside"):
                    _normalize_source_dir(local_model, label="Model directory")

    def test_quantize_model_uses_real_pipeline_with_external_binary_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "source-model"
            local_root = root / "Local"
            model_dir.mkdir()
            _write_config(model_dir)
            _write_safetensors(
                model_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": (
                        "F32",
                        [100, 128],
                        b"\0" * 100 * 128 * 4,
                    )
                },
            )
            with patch.object(_model_store, "LOCAL_ROOT", local_root), patch(
                "trillim.quantize._entrypoint.resolve_quantize_binary",
                return_value=Path("/bin/true"),
            ):
                result = quantize(model_dir)

            self.assertEqual(result.bundle_type, "model")
            self.assertFalse(result.used_language_model_only)
            self.assertEqual(result.bundle_path, local_root / "source-model-TRNQ")
            self.assertTrue((result.bundle_path / "trillim_config.json").is_file())
            self.assertTrue((result.bundle_path / "config.json").is_file())
