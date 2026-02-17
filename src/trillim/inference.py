# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

from trillim.model_arch import ModelConfig as ArchConfig
from prompt_toolkit import prompt as better_input
from trillim.token_utils import IncrementalDecoder
from transformers import AutoTokenizer


def _load_from_path(model_path: str, trust_remote_code: bool = False):
    """
    Load tokenizer from a single path, handling custom tokenizer classes.
    If tokenizer_config.json specifies a custom tokenizer_class (e.g. BitnetTokenizer),
    dynamically import it from the model directory — but only when trust_remote_code=True.
    """
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")

    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
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


def load_tokenizer(model_dir: str, use_lora: bool = False, trust_remote_code: bool = False):
    """
    Load tokenizer from model directory, with optional LoRA overrides.

    When use_lora=True:
    - If lora_tokenizer.json exists, creates a temp directory with merged tokenizer files
    - Otherwise, loads base tokenizer and applies overrides from lora_tokenizer_config.json
    - Loads chat template from lora_chat_template.jinja if present
    """
    lora_tokenizer_path = os.path.join(model_dir, "lora_tokenizer.json")

    if use_lora and os.path.exists(lora_tokenizer_path):
        # LoRA adapter has its own tokenizer - use a temp directory to load it
        lora_tok_dir = tempfile.mkdtemp(prefix="lora_tok_")
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
            lora_tok_cfg_path = os.path.join(model_dir, "lora_tokenizer_config.json")
            if os.path.exists(lora_tok_cfg_path):
                shutil.copy(
                    lora_tok_cfg_path,
                    os.path.join(lora_tok_dir, "tokenizer_config.json"),
                )
            lora_chat_template_path = os.path.join(
                model_dir, "lora_chat_template.jinja"
            )
            if os.path.exists(lora_chat_template_path):
                shutil.copy(
                    lora_chat_template_path,
                    os.path.join(lora_tok_dir, "chat_template.jinja"),
                )
            tokenizer = _load_from_path(lora_tok_dir, trust_remote_code=trust_remote_code)
        finally:
            shutil.rmtree(lora_tok_dir, ignore_errors=True)
    else:
        tokenizer = _load_from_path(model_dir, trust_remote_code=trust_remote_code)

        # Apply tokenizer config overrides if present
        if use_lora:
            lora_tok_cfg_path = os.path.join(model_dir, "lora_tokenizer_config.json")
            if os.path.exists(lora_tok_cfg_path):
                with open(lora_tok_cfg_path) as f:
                    lora_tok_cfg = json.load(f)
                if "chat_template" in lora_tok_cfg:
                    tokenizer.chat_template = lora_tok_cfg["chat_template"]
                if "eos_token" in lora_tok_cfg:
                    tokenizer.eos_token = lora_tok_cfg["eos_token"]
                if "bos_token" in lora_tok_cfg:
                    tokenizer.bos_token = lora_tok_cfg["bos_token"]
            # Also check for standalone chat template file
            lora_chat_template_path = os.path.join(
                model_dir, "lora_chat_template.jinja"
            )
            if os.path.exists(lora_chat_template_path):
                with open(lora_chat_template_path) as f:
                    tokenizer.chat_template = f.read()

    return tokenizer


def _config_args(arch_config, num_threads: int = 0) -> list[str]:
    """Build --config CLI args for the C++ inference binary.

    Args:
        num_threads: Thread count for inference. 0 = auto (hardware_concurrency - 2).
    """
    args = [
        "--config",
        str(int(arch_config.arch_type)),
        str(int(arch_config.arch_info.activation)),
        str(arch_config.hidden_dim),
        str(arch_config.intermediate_dim),
        str(arch_config.num_layers),
        str(arch_config.num_heads),
        str(arch_config.num_kv_heads),
        str(arch_config.vocab_size),
        str(arch_config.head_dim),
        str(arch_config.max_position_embeddings),
        str(arch_config.norm_eps),
        str(arch_config.rope_theta),
        "1" if arch_config.tie_word_embeddings else "0",
        "1" if arch_config.has_qkv_bias else "0",
        "1" if arch_config.arch_info.has_attn_sub_norm else "0",
        "1" if arch_config.arch_info.has_ffn_sub_norm else "0",
        str(num_threads),
        str(len(arch_config.eos_tokens)),
    ] + [str(t) for t in arch_config.eos_tokens]
    return args


def main():
    if len(sys.argv) < 2:
        print("Usage: trillim chat <model_directory> [--lora] [--threads N]")
        sys.exit(1)

    MODEL_PATH = sys.argv[1].strip()
    if len(MODEL_PATH) > 1 and MODEL_PATH[-1] == "/":
        MODEL_PATH = MODEL_PATH[:-1]

    USE_LORA = "--lora" in sys.argv
    TRUST_REMOTE_CODE = "--trust-remote-code" in sys.argv
    num_threads = 0
    if "--threads" in sys.argv:
        idx = sys.argv.index("--threads")
        if idx + 1 < len(sys.argv):
            num_threads = int(sys.argv[idx + 1])
    config_path = os.path.join(MODEL_PATH, "config.json")

    if USE_LORA:
        lora_path = os.path.join(MODEL_PATH, "qmodel.lora")
        if not os.path.exists(lora_path):
            print(
                f"Error: --lora flag set but {lora_path} not found. "
                "Run 'make generate MODEL_DIR=<path> ADAPTER_DIR=<path>' first."
            )
            sys.exit(1)

    try:
        tokenizer = load_tokenizer(MODEL_PATH, USE_LORA, trust_remote_code=TRUST_REMOTE_CODE)

        arch_config = ArchConfig.from_config_json(config_path, MODEL_PATH)
        stop_tokens = set(arch_config.eos_tokens)

        from trillim._bin_path import inference_bin

        cmd = [inference_bin(), MODEL_PATH] + _config_args(arch_config, num_threads=num_threads)
        if USE_LORA:
            cmd.append("--lora")
        model = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        max_context = arch_config.max_position_embeddings
        has_chat_template = (
            hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        )
        messages = []
        cached_token_ids = []
        cached_prompt_str = ""

        MODEL_NAME = MODEL_PATH[MODEL_PATH.rfind('/')+1:]
        print(f"Talk to {MODEL_NAME} (Ctrl+D or 'q' to quit, '/new' for new conversation)")
        while True:
            try:
                query = better_input("> ")
            except (EOFError, KeyboardInterrupt):
                query = "q"

            if query.strip() == "q":
                model.stdin.write("0\n")
                model.stdin.flush()
                break

            if query.strip() == "/new":
                messages = []
                cached_token_ids = []
                cached_prompt_str = ""
                print("Starting new conversation.")
                continue

            messages.append({"role": "user", "content": query})

            if has_chat_template:
                new_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if cached_prompt_str:
                    # Incremental: only encode the new suffix
                    suffix_str = new_prompt[len(cached_prompt_str) :]
                    delta_tokens = tokenizer.encode(
                        suffix_str, add_special_tokens=False
                    )
                    all_token_ids = cached_token_ids + delta_tokens
                    reset_flag = 0
                else:
                    # First turn or after reset: encode full prompt
                    all_token_ids = tokenizer.encode(
                        new_prompt, add_special_tokens=False
                    )
                    delta_tokens = all_token_ids
                    reset_flag = 1
            else:
                all_token_ids = tokenizer.encode(query)
                delta_tokens = all_token_ids
                reset_flag = 1

            # Context limit check
            if len(all_token_ids) >= max_context:
                print(
                    f"Context window full ({max_context} tokens). Starting new conversation."
                )
                messages = [messages[-1]]
                cached_token_ids = []
                cached_prompt_str = ""
                if has_chat_template:
                    new_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    all_token_ids = tokenizer.encode(
                        new_prompt, add_special_tokens=False
                    )
                else:
                    all_token_ids = tokenizer.encode(query)
                delta_tokens = all_token_ids
                reset_flag = 1

            # Send protocol: num_tokens, reset_flag, sampling params, token IDs
            model.stdin.write(f"{len(delta_tokens)}\n")
            model.stdin.write(f"{reset_flag}\n")
            model.stdin.flush()

            # temperature top_k top_p repetition_penalty rep_penalty_lookback max_tokens
            model.stdin.write("0.6\n50\n0.9\n1.1\n64\n0\n")
            model.stdin.flush()

            for tok in delta_tokens:
                model.stdin.write(f"{tok}\n")
                model.stdin.flush()

            print("Model Response: ", end="", flush=True)
            decoder = IncrementalDecoder(tokenizer)
            generated_tokens = []
            response_text = ""
            while True:
                out = model.stdout.readline()
                if not out:
                    break
                token = int(out.strip())
                if token in stop_tokens:
                    generated_tokens.append(token)
                    break

                generated_tokens.append(token)
                new_text = decoder.decode(token)
                response_text += new_text
                print(new_text, end="", flush=True)
            print()

            # Read kv_position line
            kv_line = model.stdout.readline()
            if kv_line:
                kv_position = int(kv_line.strip())
                cached_token_ids = (all_token_ids + generated_tokens)[:kv_position]

            # Update cached prompt string and message history
            messages.append({"role": "assistant", "content": response_text})
            if has_chat_template:
                cached_prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

    except BrokenPipeError:
        stderr = ""
        try:
            stderr = model.stderr.read() if model and model.stderr else ""
        except Exception:
            pass
        print("\nError: Inference engine crashed.")
        if stderr:
            print(stderr.strip())
        print("\nIf you think the engine is broken, please report the bug!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
