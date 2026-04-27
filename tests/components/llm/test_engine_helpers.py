from __future__ import annotations

import unittest
from pathlib import Path

from trillim.components.llm._config import (
    ActivationType,
    ArchitectureType,
    InitConfig,
    ModelRuntimeConfig,
    SamplingDefaults,
)
from trillim.components.llm._engine import (
    EngineProtocolError,
    EngineCrashedError,
    InferenceEngine,
    _bundled_binary_path,
    _build_init_block,
    _build_request_block,
    _common_prefix_len,
    _first_protocol_line,
    _parse_protocol_int,
    _read_stderr,
)


def _model() -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        name="model",
        path=Path("/tmp/model"),
        arch_type=ArchitectureType.LLAMA,
        activation=ActivationType.SILU,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=100,
        head_dim=64,
        max_position_embeddings=128,
        norm_eps=1e-6,
        rope_theta=10000.0,
        eos_tokens=(2, 3),
        has_qkv_bias=True,
        tie_word_embeddings=False,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=True,
    )


class EngineHelperTests(unittest.TestCase):
    def test_build_init_block_serializes_model_and_optional_init_config(self):
        block = _build_init_block(
            _model(),
            InitConfig(
                model_dir=Path("/tmp/model"),
                num_threads=4,
                lora_dir=Path("/tmp/lora\nignored"),
                lora_quant="q8\nignored",
                unembed_quant="q4",
            ),
        )

        lines = block.splitlines()
        self.assertEqual(int(lines[0]), len(lines) - 1)
        self.assertIn("arch_type=2", lines)
        self.assertIn("activation=1", lines)
        self.assertIn("eos_tokens=2,3", lines)
        self.assertIn("num_threads=4", lines)
        self.assertIn("lora_dir=/tmp/lora", lines)
        self.assertIn("lora_quant=q8", lines)
        self.assertIn("unembed_quant=q4", lines)

    def test_build_request_block_serializes_sampling_and_tokens(self):
        block = _build_request_block(
            kv_position=2,
            tokens=[10, 11],
            temperature=0.5,
            top_k=3,
            top_p=0.9,
            repetition_penalty=1.1,
            rep_penalty_lookback=8,
            max_tokens=16,
        )

        self.assertEqual(
            block,
            "8\nkv_position=2\ntokens=10,11\ntemperature=0.5\ntop_k=3\n"
            "top_p=0.9\nrepetition_penalty=1.1\nrep_penalty_lookback=8\n"
            "max_tokens=16\n",
        )

    def test_protocol_helpers_handle_valid_and_invalid_values(self):
        self.assertEqual(_first_protocol_line("  a\nb"), "a")
        self.assertEqual(_first_protocol_line("   "), "   ")
        self.assertEqual(_common_prefix_len([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(_parse_protocol_int(b" 42\n", "token_id"), 42)

        with self.assertRaisesRegex(EngineProtocolError, "expected int"):
            _parse_protocol_int(b"nope", "token_id")

    def test_inference_engine_requires_running_process_and_stop_is_idempotent(self):
        import asyncio

        async def run() -> None:
            engine = InferenceEngine(
                _model(),
                tokenizer=object(),
                defaults=SamplingDefaults(),
                progress_timeout=0.01,
            )
            self.assertTrue(Path(_bundled_binary_path()).is_file())
            with self.assertRaisesRegex(EngineCrashedError, "not running"):
                engine._require_running()
            await engine.stop()

        asyncio.run(run())

    def test_read_stderr_decodes_real_subprocess_stderr(self):
        import asyncio
        import sys

        async def run() -> str:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('boom')",
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            return await _read_stderr(process)

        self.assertEqual(asyncio.run(run()), "boom")
