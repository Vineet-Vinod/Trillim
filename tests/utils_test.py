# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for shared utility helpers."""

import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trillim import utils


class UtilsTests(unittest.TestCase):
    def test_load_from_path_falls_back_to_auto_tokenizer_with_trust_remote_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("trillim.utils.AutoTokenizer.from_pretrained", return_value="tokenizer") as mock_auto:
                tokenizer = utils._load_from_path(temp_dir, trust_remote_code=True)

        self.assertEqual(tokenizer, "tokenizer")
        mock_auto.assert_called_once_with(temp_dir, trust_remote_code=True)

    def test_load_from_path_warns_and_falls_back_without_trust_remote_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "FancyTokenizer"}),
                encoding="utf-8",
            )
            (model_dir / "tokenization_fancy.py").write_text(
                "class FancyTokenizer:\n    pass\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with (
                contextlib.redirect_stderr(stderr),
                patch("trillim.utils.AutoTokenizer.from_pretrained", return_value="fallback") as mock_auto,
            ):
                tokenizer = utils._load_from_path(temp_dir, trust_remote_code=False)

        self.assertEqual(tokenizer, "fallback")
        self.assertIn("Re-run with --trust-remote-code", stderr.getvalue())
        mock_auto.assert_called_once_with(temp_dir, trust_remote_code=False)

    def test_load_from_path_loads_custom_tokenizer_when_trusted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "FancyTokenizer"}),
                encoding="utf-8",
            )
            (model_dir / "tokenization_fancy.py").write_text(
                "class FancyTokenizer:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, model_path):\n"
                "        return {'loaded_from': model_path}\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with (
                contextlib.redirect_stderr(stderr),
                patch("trillim.utils.AutoTokenizer.from_pretrained") as mock_auto,
            ):
                tokenizer = utils._load_from_path(temp_dir, trust_remote_code=True)

        self.assertEqual(tokenizer, {"loaded_from": temp_dir})
        self.assertIn("Loading custom tokenizer code", stderr.getvalue())
        mock_auto.assert_not_called()

    def test_load_from_path_falls_back_when_custom_class_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "FancyTokenizer"}),
                encoding="utf-8",
            )
            (model_dir / "tokenization_fancy.py").write_text(
                "class DifferentTokenizer:\n    pass\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()

            with (
                contextlib.redirect_stderr(stderr),
                patch("trillim.utils.AutoTokenizer.from_pretrained", return_value="fallback") as mock_auto,
            ):
                tokenizer = utils._load_from_path(temp_dir, trust_remote_code=True)

        self.assertEqual(tokenizer, "fallback")
        self.assertIn("Loading custom tokenizer code", stderr.getvalue())
        mock_auto.assert_called_once_with(temp_dir, trust_remote_code=True)

    def test_load_from_path_ignores_standard_tokenizer_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
                encoding="utf-8",
            )

            with patch("trillim.utils.AutoTokenizer.from_pretrained", return_value="fast") as mock_auto:
                tokenizer = utils._load_from_path(temp_dir)

        self.assertEqual(tokenizer, "fast")
        mock_auto.assert_called_once_with(temp_dir, trust_remote_code=False)

    def test_load_tokenizer_uses_temporary_dir_for_lora_tokenizer_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenization_fancy.py").write_text("# tokenization\n", encoding="utf-8")
            (model_dir / "spiece.model").write_text("model", encoding="utf-8")
            (adapter_dir / "lora_tokenizer.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "lora_tokenizer_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "lora_chat_template.jinja").write_text("{{ prompt }}", encoding="utf-8")
            seen: dict[str, object] = {}

            def fake_load(path: str, trust_remote_code: bool = False):
                temp_path = Path(path)
                seen["path"] = temp_path
                seen["exists_during"] = temp_path.exists()
                seen["files"] = sorted(p.name for p in temp_path.iterdir())
                seen["trust_remote_code"] = trust_remote_code
                return "merged-tokenizer"

            with patch("trillim.utils._load_from_path", side_effect=fake_load):
                tokenizer = utils.load_tokenizer(
                    str(model_dir),
                    adapter_dir=str(adapter_dir),
                    trust_remote_code=True,
                )

        self.assertEqual(tokenizer, "merged-tokenizer")
        self.assertTrue(seen["exists_during"])
        self.assertEqual(
            seen["files"],
            [
                "chat_template.jinja",
                "config.json",
                "spiece.model",
                "tokenization_fancy.py",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
        self.assertTrue(seen["trust_remote_code"])
        self.assertFalse(seen["path"].exists())

    def test_load_tokenizer_applies_lora_overrides_without_lora_tokenizer_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            adapter_dir = root / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            (adapter_dir / "lora_tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "chat_template": "config template",
                        "eos_token": "<eos>",
                        "bos_token": "<bos>",
                    }
                ),
                encoding="utf-8",
            )
            (adapter_dir / "lora_chat_template.jinja").write_text(
                "standalone template",
                encoding="utf-8",
            )
            tokenizer = SimpleNamespace(chat_template=None, eos_token=None, bos_token=None)

            with patch("trillim.utils._load_from_path", return_value=tokenizer) as mock_load:
                loaded = utils.load_tokenizer(
                    str(model_dir),
                    adapter_dir=str(adapter_dir),
                    trust_remote_code=True,
                )

        self.assertIs(loaded, tokenizer)
        mock_load.assert_called_once_with(str(model_dir), trust_remote_code=True)
        self.assertEqual(tokenizer.chat_template, "standalone template")
        self.assertEqual(tokenizer.eos_token, "<eos>")
        self.assertEqual(tokenizer.bos_token, "<bos>")

    def test_load_default_params_uses_generation_config_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "generation_config.json").write_text(
                json.dumps(
                    {
                        "temperature": 0.2,
                        "top_k": 10,
                        "top_p": 0.7,
                        "repetition_penalty": 1.5,
                        "rep_penalty_lookback": 12,
                        "ignored": 99,
                    }
                ),
                encoding="utf-8",
            )

            params = utils.load_default_params(temp_dir)

        self.assertEqual(
            params,
            {
                "temperature": 0.2,
                "top_k": 10,
                "top_p": 0.7,
                "repetition_penalty": 1.5,
                "rep_penalty_lookback": 12,
            },
        )

    def test_load_default_params_uses_defaults_when_config_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            params = utils.load_default_params(temp_dir)

        self.assertEqual(
            params,
            {
                "temperature": 0.6,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "rep_penalty_lookback": 64,
            },
        )

    def test_load_engine_options_omits_defaults_and_includes_overrides(self):
        self.assertEqual(utils.load_engine_options(), {})
        self.assertEqual(
            utils.load_engine_options(
                num_threads=4,
                lora_quant="q4",
                unembed_quant="q8",
            ),
            {
                "num_threads": 4,
                "lora_quant": "q4",
                "unembed_quant": "q8",
            },
        )

    def test_build_init_config_includes_optional_fields_only_when_set(self):
        arch_config = SimpleNamespace(
            arch_type=1,
            arch_info=SimpleNamespace(
                activation=2,
                has_attn_sub_norm=True,
                has_ffn_sub_norm=False,
            ),
            hidden_dim=128,
            intermediate_dim=256,
            num_layers=4,
            num_heads=8,
            num_kv_heads=2,
            vocab_size=32000,
            head_dim=16,
            max_position_embeddings=4096,
            norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            has_qkv_bias=False,
            eos_tokens=[2, 3],
        )

        block = utils._build_init_config(
            arch_config,
            adapter_dir="/adapter",
            num_threads=8,
            lora_quant="q4",
            unembed_quant="q8",
        )

        self.assertTrue(block.startswith("21\n"))
        self.assertIn("arch_type=1\n", block)
        self.assertIn("activation=2\n", block)
        self.assertIn("eos_tokens=2,3\n", block)
        self.assertIn("lora_dir=/adapter\n", block)
        self.assertIn("num_threads=8\n", block)
        self.assertIn("lora_quant=q4\n", block)
        self.assertIn("unembed_quant=q8\n", block)

        minimal = utils._build_init_config(arch_config)
        self.assertTrue(minimal.startswith("17\n"))
        self.assertNotIn("lora_dir=", minimal)
        self.assertNotIn("num_threads=", minimal)

    def test_build_request_block_formats_optional_sampling_params(self):
        block = utils._build_request_block(
            [1, 2, 3],
            1,
            temperature=0.1,
            top_k=5,
            top_p=0.8,
            repetition_penalty=1.2,
            rep_penalty_lookback=16,
            max_tokens=32,
        )

        self.assertTrue(block.startswith("8\n"))
        self.assertIn("reset=1\n", block)
        self.assertIn("tokens=1,2,3\n", block)
        self.assertIn("temperature=0.1\n", block)
        self.assertIn("max_tokens=32\n", block)

        minimal = utils._build_request_block([], 0)
        self.assertEqual(minimal, "2\nreset=0\ntokens=\n")

        zero_max = utils._build_request_block([], 0, max_tokens=0)
        self.assertEqual(zero_max, "2\nreset=0\ntokens=\n")

    def test_build_request_block_rejects_invalid_sampling_values(self):
        with self.assertRaisesRegex(ValueError, "temperature must be >= 0"):
            utils._build_request_block([], 0, temperature=-0.1)

        with self.assertRaisesRegex(ValueError, "max_tokens must be >= 0"):
            utils._build_request_block([], 0, max_tokens=-1)

    def test_compute_base_model_hash_handles_missing_invalid_and_valid_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(utils.compute_base_model_hash(temp_dir), "")

            model_dir = Path(temp_dir)
            config_path = model_dir / "config.json"
            config_path.write_text("{invalid", encoding="utf-8")
            self.assertEqual(utils.compute_base_model_hash(temp_dir), "")

            config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["MyModel"],
                        "hidden_size": 256,
                        "intermediate_size": 512,
                        "num_hidden_layers": 6,
                        "num_attention_heads": 8,
                        "num_key_value_heads": 2,
                        "vocab_size": 32000,
                        "ignored": "field",
                    }
                ),
                encoding="utf-8",
            )
            first_hash = utils.compute_base_model_hash(temp_dir)
            second_hash = utils.compute_base_model_hash(temp_dir)

        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)
        self.assertRegex(first_hash, r"^[0-9a-f]{64}$")

    def test_compute_base_model_hash_uses_nested_text_config_for_qwen35(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "model_type": "qwen3_5",
                        "hidden_size": 1,
                        "text_config": {
                            "hidden_size": 2560,
                            "intermediate_size": 9216,
                            "num_hidden_layers": 32,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 4,
                            "vocab_size": 248320,
                        },
                    }
                ),
                encoding="utf-8",
            )

            first_hash = utils.compute_base_model_hash(temp_dir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "model_type": "qwen3_5",
                        "hidden_size": 9999,
                        "text_config": {
                            "hidden_size": 2560,
                            "intermediate_size": 9216,
                            "num_hidden_layers": 32,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 4,
                            "vocab_size": 248320,
                        },
                    }
                ),
                encoding="utf-8",
            )
            second_hash = utils.compute_base_model_hash(temp_dir)

        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)


if __name__ == "__main__":
    unittest.main()
