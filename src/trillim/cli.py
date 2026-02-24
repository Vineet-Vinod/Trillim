# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unified CLI entry point for Trillim."""

import argparse
import json
import os
import sys

from collections import defaultdict


def _resolve(args):
    """Resolve model_dir if it looks like a HuggingFace model ID."""
    if hasattr(args, "model_dir"):
        from trillim.model_store import resolve_model_dir

        args.model_dir = resolve_model_dir(args.model_dir)


def _cmd_quantize(args):
    """Run standalone quantization."""
    try:
        _resolve(args)
        argv = [sys.argv[0], args.model_dir]
        if args.model:
            argv.append("--model")
        if args.adapter:
            argv.extend(["--adapter", args.adapter])
        sys.argv = argv

        from trillim.quantize import main

        main()
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_chat(args):
    """Run interactive chat."""
    try:
        _resolve(args)
        argv = [sys.argv[0], args.model_dir]
        if args.lora:
            from trillim.model_store import resolve_model_dir

            argv.extend(["--lora", resolve_model_dir(args.lora)])
        if args.threads:
            argv.extend(["--threads", str(args.threads)])
        if args.lora_quant:
            argv.extend(["--lora-quant", args.lora_quant])
        if args.unembed_quant:
            argv.extend(["--unembed-quant", args.unembed_quant])
        if args.trust_remote_code:
            argv.append("--trust-remote-code")
        sys.argv = argv

        from trillim.inference import main

        main()
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_serve(args):
    """Start the API server."""
    try:
        _resolve(args)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from trillim import LLM, Server, TTS, Whisper

        components = [LLM(args.model_dir, num_threads=args.threads or 0, trust_remote_code=args.trust_remote_code, lora_quant=args.lora_quant, unembed_quant=args.unembed_quant)]
        if args.voice:
            components.append(Whisper(model_size=args.whisper_model))
            components.append(TTS(voices_dir=args.voices_dir))

        Server(*components).run(host=args.host, port=args.port)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_pull(args):
    """Pull a pre-quantized model from HuggingFace."""
    try:
        from trillim.model_store import pull_model

        pull_model(
            args.model_id,
            revision=args.revision,
            token=args.token,
            force=args.force,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_models(args):
    """List locally downloaded models and adapters."""
    from trillim.model_store import list_adapters, list_models

    models = list_models()
    adapters = list_adapters()

    if args.json:
        print(json.dumps({"models": models, "adapters": adapters}, indent=2))
        return

    if not models and not adapters:
        print("No models found. Run: trillim pull <org/model>")
        return

    if models:
        # Column widths
        id_w = max(len(m["model_id"]) for m in models)
        id_w = max(id_w, len("MODEL ID"))
        arch_w = 10
        size_w = 8

        print("Models")
        header = (
            f"{'MODEL ID':<{id_w}}  {'ARCH':<{arch_w}}  {'SIZE':>{size_w}}  {'SOURCE'}"
        )
        sep = f"{'-' * id_w}  {'-' * arch_w}  {'-' * size_w}  {'-' * 30}"
        print(header)
        print(sep)
        for m in models:
            arch = m.get("architecture", "")[:arch_w]
            size = m.get("size_human", "-")
            source = m.get("source_model", "")
            print(f"{m['model_id']:<{id_w}}  {arch:<{arch_w}}  {size:>{size_w}}  {source}")

    if adapters:
        if models:
            print()

        # Build hash → model ID(s) mapping for compatibility display
        hash_to_models: defaultdict[str, list[str]] = defaultdict()
        for m in models:
            h = m.get("base_model_config_hash", "")
            hash_to_models[h].append(m["model_id"])

        id_w = max(len(a["model_id"]) for a in adapters)
        id_w = max(id_w, len("ADAPTER ID"))
        size_w = 8

        print("Adapters")
        header = f"{'ADAPTER ID':<{id_w}}  {'SIZE':>{size_w}}  {'COMPATIBLE MODELS'}"
        sep = f"{'-' * id_w}  {'-' * size_w}  {'-' * 30}"
        print(header)
        print(sep)
        pad = f"{'':<{id_w}}  {'':>{size_w}}  "
        for a in adapters:
            size = a.get("size_human", "-")
            adapter_hash = a.get("base_model_config_hash", "")
            compat = hash_to_models[adapter_hash] if adapter_hash else []
            first_line = compat[0] if compat else "(none)"
            print(f"{a['model_id']:<{id_w}}  {size:>{size_w}}  {first_line}")
            for extra in compat[1:]:
                print(f"{pad}{extra}")


def main():
    parser = argparse.ArgumentParser(
        prog="trillim",
        description="Trillim — the fastest CPU inference engine for BitNet models",
    )
    sub = parser.add_subparsers(dest="command")

    # --- quantize ---
    p_quant = sub.add_parser("quantize", help="Quantize safetensors and/or extract LoRA adapter")
    p_quant.add_argument("model_dir", help="Path to model directory with config.json")
    p_quant.add_argument("--model", action="store_true", help="Quantize model weights → <model_dir>-TRNQ/")
    p_quant.add_argument("--adapter", help="Extract LoRA adapter from PEFT directory → qmodel.lora")
    p_quant.set_defaults(func=_cmd_quantize)

    # --- chat ---
    p_chat = sub.add_parser("chat", help="Interactive chat with a model")
    p_chat.add_argument("model_dir", help="Path to model directory or HuggingFace model ID")
    p_chat.add_argument("--lora", help="Path to LoRA adapter directory")
    p_chat.add_argument("--threads", type=int, default=0, help="Number of threads (0 = auto)")
    p_chat.add_argument("--lora-quant", default=None, help="LoRA quantization (none|int8|q4_0|q5_0|q6_k|q8_0)")
    p_chat.add_argument("--unembed-quant", default=None, help="Unembed quantization (int8|q4_0|q5_0|q6_k|q8_0)")
    p_chat.add_argument("--trust-remote-code", action="store_true", help="Allow loading custom tokenizer code from model directory")
    p_chat.set_defaults(func=_cmd_chat)

    # --- serve ---
    from pathlib import Path

    _default_voices_dir = str(Path.home() / ".trillim" / "voices")

    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    p_serve.add_argument("model_dir", help="Path to model directory or HuggingFace model ID")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to bind to")
    p_serve.add_argument(
        "--voice", action="store_true", help="Enable voice pipeline (STT + TTS)"
    )
    p_serve.add_argument(
        "--whisper-model", default="base.en", help="Whisper model size (default: base.en)"
    )
    p_serve.add_argument(
        "--voices-dir",
        default=_default_voices_dir,
        help="Directory for persistent custom voice WAVs (default: ~/.trillim/voices)",
    )
    p_serve.add_argument("--threads", type=int, default=0, help="Number of threads (0 = auto)")
    p_serve.add_argument("--lora-quant", default=None, help="LoRA quantization (none|int8|q4_0|q5_0|q6_k|q8_0)")
    p_serve.add_argument("--unembed-quant", default=None, help="Unembed quantization (int8|q4_0|q5_0|q6_k|q8_0)")
    p_serve.add_argument("--trust-remote-code", action="store_true", help="Allow loading custom tokenizer code from model directory")
    p_serve.set_defaults(func=_cmd_serve)

    # --- pull ---
    p_pull = sub.add_parser("pull", help="Download a pre-quantized model from HuggingFace")
    p_pull.add_argument("model_id", help="HuggingFace model ID (e.g. Trillim/BitNet-3B-TRNQ)")
    p_pull.add_argument("--revision", help="Branch, tag, or commit hash")
    p_pull.add_argument("--token", help="HuggingFace token for gated/private repos")
    p_pull.add_argument("--force", "-f", action="store_true", help="Re-download even if exists")
    p_pull.set_defaults(func=_cmd_pull)

    # --- models ---
    p_models = sub.add_parser("models", help="List locally downloaded models")
    p_models.add_argument("--json", action="store_true", help="Output as JSON")
    p_models.set_defaults(func=_cmd_models)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
