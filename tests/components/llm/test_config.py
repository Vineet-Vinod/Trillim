"""Tests for LLM configuration helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trillim.components.llm._config import (
    ActivationType,
    ArchitectureType,
    LLMState,
    ModelInfo,
    SamplingDefaults,
    load_sampling_defaults,
)


class LLMConfigTests(unittest.TestCase):
    def test_enums_and_model_info_are_stable(self):
        info = ModelInfo(LLMState.RUNNING, "model", "/tmp/model", 123, False)

        self.assertEqual(info.state, LLMState.RUNNING)
        self.assertEqual(LLMState.SERVER_ERROR.value, "server_error")
        self.assertEqual(ArchitectureType.LLAMA, 2)
        self.assertEqual(ActivationType.SILU, 1)

    def test_load_sampling_defaults_uses_generation_config_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "generation_config.json").write_text(
                json.dumps(
                    {
                        "temperature": 0.2,
                        "top_k": 12,
                        "top_p": 0.5,
                        "repetition_penalty": 1.4,
                        "max_new_tokens": 99,
                    }
                ),
                encoding="utf-8",
            )

            defaults = load_sampling_defaults(root)

        self.assertEqual(
            defaults,
            SamplingDefaults(
                temperature=0.2,
                top_k=12,
                top_p=0.5,
                repetition_penalty=1.4,
                max_tokens=99,
            ),
        )
