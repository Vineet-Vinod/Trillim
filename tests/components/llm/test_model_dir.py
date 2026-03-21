"""Tests for model-directory validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trillim.components.llm._config import ActivationType, ArchitectureType
from trillim.components.llm._model_dir import validate_model_dir
from trillim.errors import ModelValidationError
from tests.components.llm.support import model_dir


class ModelDirectoryTests(unittest.TestCase):
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
