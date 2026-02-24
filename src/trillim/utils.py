# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Shared utility functions used across inference, quantization, and the API server."""

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile

from transformers import AutoTokenizer


def _load_from_path(model_path: str, trust_remote_code: bool = False):
    """
    Load tokenizer from a single path, handling custom tokenizer classes.
    If tokenizer_config.json specifies a custom tokenizer_class (e.g. BitnetTokenizer),
    dynamically import it from the model directory — but only when trust_remote_code=True.
    """
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")

    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        tokenizer_class = tokenizer_config.get("tokenizer_class", "")

        # Check for custom tokenizer classes that need to be imported from model dir
        if tokenizer_class and tokenizer_class not in (
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ):
            # Look for a tokenization_*.py file in the model directory
            tokenization_file = os.path.join(
                model_path,
                f"tokenization_{tokenizer_class.lower().replace('tokenizer', '')}.py",
            )

            if os.path.exists(tokenization_file):
                if not trust_remote_code:
                    print(
                        f"WARNING: This model requires custom tokenizer code ({os.path.basename(tokenization_file)}). "
                        "Re-run with --trust-remote-code to allow loading it. "
                        "Falling back to AutoTokenizer.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"WARNING: Loading custom tokenizer code ({os.path.basename(tokenization_file)}) "
                        "from model directory. Only use --trust-remote-code with models you trust.",
                        file=sys.stderr,
                    )
                    spec = importlib.util.spec_from_file_location(
                        "custom_tokenizer", tokenization_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, tokenizer_class):
                        custom_tokenizer_cls = getattr(module, tokenizer_class)
                        return custom_tokenizer_cls.from_pretrained(model_path)

    # Fall back to AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path)


def load_tokenizer(model_dir: str, adapter_dir: str | None = None, trust_remote_code: bool = False):
    """
    Load tokenizer from model directory, with optional LoRA overrides.

    When adapter_dir is set:
    - Looks for lora_* tokenizer files in adapter_dir (not model_dir)
    - If lora_tokenizer.json exists, creates a temp directory with merged tokenizer files
    - Otherwise, loads base tokenizer and applies overrides from lora_tokenizer_config.json
    - Loads chat template from lora_chat_template.jinja if present
    """
    # Directory to look for lora_* files — adapter_dir if separate, else model_dir
    lora_dir = adapter_dir or model_dir
    lora_tokenizer_path = os.path.join(lora_dir, "lora_tokenizer.json")

    if adapter_dir and os.path.exists(lora_tokenizer_path):
        # LoRA adapter has its own tokenizer - use a temp directory to load it
        lora_tok_dir = tempfile.mkdtemp(prefix="lora_tok_")
        os.chmod(lora_tok_dir, 0o700)
        try:
            # Copy base model files needed by transformers
            for fname in os.listdir(model_dir):
                if (
                    fname.startswith("tokenization_")
                    or fname.endswith(".model")
                    or fname in ("config.json", "tokenizer_config.json")
                ):
                    shutil.copy(os.path.join(model_dir, fname), lora_tok_dir)
            # Copy LoRA tokenizer files (without lora_ prefix)
            shutil.copy(
                lora_tokenizer_path, os.path.join(lora_tok_dir, "tokenizer.json")
            )
            lora_tok_cfg_path = os.path.join(lora_dir, "lora_tokenizer_config.json")
            if os.path.exists(lora_tok_cfg_path):
                shutil.copy(
                    lora_tok_cfg_path,
                    os.path.join(lora_tok_dir, "tokenizer_config.json"),
                )
            lora_chat_template_path = os.path.join(
                lora_dir, "lora_chat_template.jinja"
            )
            if os.path.exists(lora_chat_template_path):
                shutil.copy(
                    lora_chat_template_path,
                    os.path.join(lora_tok_dir, "chat_template.jinja"),
                )
            tokenizer = _load_from_path(lora_tok_dir, trust_remote_code=trust_remote_code)
        finally:
            shutil.rmtree(lora_tok_dir)
    else:
        tokenizer = _load_from_path(model_dir, trust_remote_code=trust_remote_code)

        # Apply tokenizer config overrides if present
        if adapter_dir:
            lora_tok_cfg_path = os.path.join(lora_dir, "lora_tokenizer_config.json")
            if os.path.exists(lora_tok_cfg_path):
                with open(lora_tok_cfg_path, encoding="utf-8") as f:
                    lora_tok_cfg = json.load(f)
                if "chat_template" in lora_tok_cfg:
                    tokenizer.chat_template = lora_tok_cfg["chat_template"]
                if "eos_token" in lora_tok_cfg:
                    tokenizer.eos_token = lora_tok_cfg["eos_token"]
                if "bos_token" in lora_tok_cfg:
                    tokenizer.bos_token = lora_tok_cfg["bos_token"]
            # Also check for standalone chat template file
            lora_chat_template_path = os.path.join(
                lora_dir, "lora_chat_template.jinja"
            )
            if os.path.exists(lora_chat_template_path):
                with open(lora_chat_template_path, encoding="utf-8") as f:
                    tokenizer.chat_template = f.read()

    return tokenizer


def load_default_params(model_dir: str) -> dict:
    """Load sampling params from generation_config.json, falling back to defaults."""
    defaults = {
        "temperature": 0.6,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "rep_penalty_lookback": 64,
    }
    gen_config_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path, encoding="utf-8") as f:
            gen_config = json.load(f)
        for key in defaults:
            if key in gen_config:
                defaults[key] = gen_config[key]
    return defaults


def load_engine_options(
    num_threads: int = 0,
    lora_quant: str | None = None,
    unembed_quant: str | None = None,
) -> dict:
    """Build engine option dict from CLI args.  Only non-default values included."""
    opts: dict = {}
    if num_threads:
        opts["num_threads"] = num_threads
    if lora_quant is not None:
        opts["lora_quant"] = lora_quant
    if unembed_quant is not None:
        opts["unembed_quant"] = unembed_quant
    return opts


def _build_init_config(
    arch_config,
    adapter_dir: str | None = None,
    num_threads: int = 0,
    lora_quant: str | None = None,
    unembed_quant: str | None = None,
) -> str:
    """Build the count-prefixed init block sent via stdin after launch.

    Always emits arch fields + eos_tokens.  Only emits ``lora_dir``,
    ``num_threads``, ``lora_quant``, and ``unembed_quant`` when non-default.
    """
    pairs = [
        f"arch_type={int(arch_config.arch_type)}",
        f"activation={int(arch_config.arch_info.activation)}",
        f"hidden_dim={arch_config.hidden_dim}",
        f"intermediate_dim={arch_config.intermediate_dim}",
        f"num_layers={arch_config.num_layers}",
        f"num_heads={arch_config.num_heads}",
        f"num_kv_heads={arch_config.num_kv_heads}",
        f"vocab_size={arch_config.vocab_size}",
        f"head_dim={arch_config.head_dim}",
        f"max_position_embeddings={arch_config.max_position_embeddings}",
        f"norm_eps={arch_config.norm_eps}",
        f"rope_theta={arch_config.rope_theta}",
        f"tie_word_embeddings={'1' if arch_config.tie_word_embeddings else '0'}",
        f"has_qkv_bias={'1' if arch_config.has_qkv_bias else '0'}",
        f"has_attn_sub_norm={'1' if arch_config.arch_info.has_attn_sub_norm else '0'}",
        f"has_ffn_sub_norm={'1' if arch_config.arch_info.has_ffn_sub_norm else '0'}",
        f"eos_tokens={','.join(str(t) for t in arch_config.eos_tokens)}",
    ]
    if adapter_dir:
        pairs.append(f"lora_dir={adapter_dir}")
    if num_threads:
        pairs.append(f"num_threads={num_threads}")
    if lora_quant is not None:
        pairs.append(f"lora_quant={lora_quant}")
    if unembed_quant is not None:
        pairs.append(f"unembed_quant={unembed_quant}")
    return f"{len(pairs)}\n" + "".join(f"{p}\n" for p in pairs)


def _build_request_block(
    delta_tokens: list[int],
    reset_flag: int,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    rep_penalty_lookback: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Build the count-prefixed per-request block.

    Always emits ``reset`` and ``tokens``.  Only emits sampling params
    when explicitly provided (not None).
    """
    pairs: list[str] = [
        f"reset={reset_flag}",
        f"tokens={','.join(str(t) for t in delta_tokens)}",
    ]
    if temperature is not None:
        pairs.append(f"temperature={temperature}")
    if top_k is not None:
        pairs.append(f"top_k={top_k}")
    if top_p is not None:
        pairs.append(f"top_p={top_p}")
    if repetition_penalty is not None:
        pairs.append(f"repetition_penalty={repetition_penalty}")
    if rep_penalty_lookback is not None:
        pairs.append(f"rep_penalty_lookback={rep_penalty_lookback}")
    if max_tokens is not None:
        pairs.append(f"max_tokens={max_tokens}")
    return f"{len(pairs)}\n" + "".join(f"{p}\n" for p in pairs)


def compute_base_model_hash(model_dir):
    """Compute a stable hash from the base model's identifying config fields.

    Returns a hex SHA-256 digest string, or "" if config.json cannot be read.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return ""
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""

    # Extract the fields that uniquely identify a model architecture.
    # Sorted keys + json.dumps(sort_keys=True) keeps the hash deterministic.
    identity = {
        "architectures": raw.get("architectures", []),
        "hidden_size": raw.get("hidden_size"),
        "intermediate_size": raw.get("intermediate_size"),
        "num_hidden_layers": raw.get("num_hidden_layers"),
        "num_attention_heads": raw.get("num_attention_heads"),
        "num_key_value_heads": raw.get("num_key_value_heads"),
        "vocab_size": raw.get("vocab_size"),
    }
    blob = json.dumps(identity, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
