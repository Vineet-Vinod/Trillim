from __future__ import annotations

import json
from pathlib import Path

from trillim._bundle_metadata import (
    CURRENT_FORMAT_VERSION,
    compute_base_model_config_hash,
)


def write_llm_bundle(
    path: Path,
    *,
    architecture: str = "LlamaForCausalLM",
    hidden_act: str = "silu",
    config_overrides: dict | None = None,
    text_config: dict | None = None,
    tokenizer_payload: dict | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": [architecture],
        "hidden_size": 129,
        "intermediate_size": 257,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32000,
        "max_position_embeddings": 512,
        "hidden_act": hidden_act,
        "eos_token_id": [2, 3],
        "rope_parameters": {"rope_theta": 12000},
        "tie_word_embeddings": True,
    }
    if config_overrides:
        config.update(config_overrides)
    if text_config is not None:
        config = {"architectures": [architecture], "text_config": text_config}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_text(
        json.dumps(
            tokenizer_payload
            or {
                "added_tokens": [
                    {"content": "</s>", "id": 2},
                    {"content": "<|im_end|>", "id": 151645},
                ]
            }
        ),
        encoding="utf-8",
    )
    (path / "qmodel.tensors").write_bytes(b"model")
    (path / "rope.cache").write_bytes(b"rope")
    (path / "trillim_config.json").write_text(
        json.dumps(
            {
                "format_version": CURRENT_FORMAT_VERSION,
                "type": "model",
                "quantization": "ternary",
                "architecture": architecture.lower(),
                "platforms": ["x86_64"],
                "source_model": "",
                "base_model_config_hash": compute_base_model_config_hash(path),
            }
        ),
        encoding="utf-8",
    )
    return path


def write_lora_bundle(adapter_dir: Path, *, model_dir: Path) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "qmodel.lora").write_bytes(b"adapter")
    (adapter_dir / "trillim_config.json").write_text(
        json.dumps(
            {
                "format_version": CURRENT_FORMAT_VERSION,
                "base_model_config_hash": compute_base_model_config_hash(model_dir),
                "source_model": "unit-test-model",
            }
        ),
        encoding="utf-8",
    )
    return adapter_dir
