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
    RuntimeInitInfo,
    SamplingDefaults,
    MAX_OUTPUT_TOKENS,
    load_sampling_defaults,
)


class LLMConfigTests(unittest.TestCase):
    def test_enums_and_model_info_are_stable(self):
        info = ModelInfo(
            LLMState.RUNNING,
            "model",
            "/tmp/model",
            123,
            False,
            adapter_path="/tmp/adapter",
            init_config=RuntimeInitInfo(
                num_threads=4,
                lora_quant="q4_0",
                unembed_quant="q8_0",
            ),
        )

        self.assertEqual(info.state, LLMState.RUNNING)
        self.assertEqual(info.adapter_path, "/tmp/adapter")
        self.assertEqual(info.init_config.num_threads, 4)
        self.assertEqual(LLMState.SERVER_ERROR.value, "server_error")
        self.assertEqual(ArchitectureType.LLAMA, 2)
        self.assertEqual(ArchitectureType.BONSAI_TERNARY, 5)
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
                        "rep_penalty_lookback": 128,
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
                rep_penalty_lookback=128,
                max_tokens=99,
            ),
        )

    def test_load_sampling_defaults_parses_rep_penalty_lookback_as_int(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "generation_config.json").write_text(
                json.dumps({"rep_penalty_lookback": 128.0}),
                encoding="utf-8",
            )

            defaults = load_sampling_defaults(root)

        self.assertEqual(defaults.rep_penalty_lookback, 128)
        self.assertIsInstance(defaults.rep_penalty_lookback, int)

    def test_load_sampling_defaults_falls_back_on_invalid_and_bool_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(load_sampling_defaults(root), SamplingDefaults())

            (root / "generation_config.json").write_text("{bad", encoding="utf-8")
            self.assertEqual(load_sampling_defaults(root), SamplingDefaults())

            (root / "generation_config.json").write_text(
                json.dumps(
                    {
                        "temperature": True,
                        "top_k": False,
                        "top_p": True,
                        "repetition_penalty": False,
                        "rep_penalty_lookback": True,
                        "max_new_tokens": False,
                    }
                ),
                encoding="utf-8",
            )
            defaults = load_sampling_defaults(root)
            self.assertEqual(defaults, SamplingDefaults())

            (root / "generation_config.json").write_text(
                json.dumps({"max_new_tokens": "not-an-int"}),
                encoding="utf-8",
            )
            self.assertEqual(load_sampling_defaults(root).max_tokens, SamplingDefaults().max_tokens)

            (root / "generation_config.json").write_text(
                json.dumps({"max_new_tokens": MAX_OUTPUT_TOKENS + 100}),
                encoding="utf-8",
            )
            self.assertEqual(load_sampling_defaults(root).max_tokens, MAX_OUTPUT_TOKENS)
