# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unified CLI entry point for Trillim."""

import argparse
import json
import os
import sys


def _resolve(args):
    """Resolve model_dir if it looks like a HuggingFace model ID."""
    if hasattr(args, "model_dir"):
        from trillim.model_store import resolve_model_dir

        args.model_dir = resolve_model_dir(args.model_dir)


def _cmd_quantize(args):
    """Run standalone quantization."""
    _resolve(args)
    argv = [sys.argv[0], args.model_dir]
    if args.model:
        argv.append("--model")
    if args.adapter:
        argv.extend(["--adapter", args.adapter])
    sys.argv = argv

    from trillim.quantize import main

    main()


def _cmd_chat(args):
    """Run interactive chat."""
    _resolve(args)
    argv = [sys.argv[0], args.model_dir]
    if args.lora:
        argv.append("--lora")
    if args.threads:
        argv.extend(["--threads", str(args.threads)])
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
    sys.argv = argv

    from trillim.inference import main

    main()


def _cmd_serve(args):
    """Start the API server."""
    _resolve(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from trillim import LLM, Server, TTS, Whisper

    components = [LLM(args.model_dir, num_threads=args.threads or 0, trust_remote_code=args.trust_remote_code)]
    if args.voice:
        components.append(Whisper(model_size=args.whisper_model))
        components.append(TTS(voices_dir=args.voices_dir))

    Server(*components).run(host=args.host, port=args.port)


def _cmd_pull(args):
    """Pull a pre-quantized model from HuggingFace."""
    from trillim.model_store import pull_model

    pull_model(
        args.model_id,
        revision=args.revision,
        token=args.token,
        force=args.force,
    )


def _cmd_models(args):
    """List locally downloaded models."""
    from trillim.model_store import list_models

    models = list_models()

    if args.json:
        print(json.dumps(models, indent=2))
        return

    if not models:
        print("No models found. Run: trillim pull <org/model>")
        return

    # Column widths
    id_w = max(len(m["model_id"]) for m in models)
    id_w = max(id_w, len("MODEL ID"))
    arch_w = 10
    size_w = 8

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


def main():
    parser = argparse.ArgumentParser(
        prog="trillim",
        description="Trillim — the fastest CPU inference engine for BitNet models",
    )
    sub = parser.add_subparsers(dest="command")

    # --- quantize ---
    p_quant = sub.add_parser("quantize", help="Quantize safetensors and/or extract LoRA adapter")
    p_quant.add_argument("model_dir", help="Path to model directory with config.json")
    p_quant.add_argument("--model", action="store_true", help="Quantize model weights (safetensors → qmodel.tensors + rope.cache)")
    p_quant.add_argument("--adapter", help="Extract LoRA adapter from PEFT directory → qmodel.lora")
    p_quant.set_defaults(func=_cmd_quantize)

    # --- chat ---
    p_chat = sub.add_parser("chat", help="Interactive chat with a model")
    p_chat.add_argument("model_dir", help="Path to model directory or HuggingFace model ID")
    p_chat.add_argument("--lora", action="store_true", help="Enable LoRA adapter")
    p_chat.add_argument("--threads", type=int, default=0, help="Number of threads (0 = auto)")
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
