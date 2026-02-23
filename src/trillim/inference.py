# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

from trillim.model_arch import ModelConfig as ArchConfig
from prompt_toolkit import prompt as better_input
from trillim.token_utils import IncrementalDecoder
from transformers import AutoTokenizer

_ENGINE_TIMEOUT = 300  # seconds; maximum wait for a single engine I/O operation


def _readline_with_timeout(pipe, timeout):
    """Read a line from a subprocess pipe with a timeout.

    Returns the line string, or None if the timeout expired.
    """
    result = []

    def _read():
        result.append(pipe.readline())

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None
    return result[0] if result else ""


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


def load_default_params() -> dict:
    """Return default sampling params."""
    return {
        "temperature": 0.6,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "rep_penalty_lookback": 64,
    }


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


def main():
    if len(sys.argv) < 2:
        print("Usage: trillim chat <model_directory> [--lora <adapter_dir>] [--threads N]")
        sys.exit(1)

    MODEL_PATH = sys.argv[1].strip()
    if len(MODEL_PATH) > 1 and MODEL_PATH[-1] == "/":
        MODEL_PATH = MODEL_PATH[:-1]

    # Parse --lora <adapter_dir>
    ADAPTER_DIR: str | None = None
    if "--lora" in sys.argv:
        lora_idx = sys.argv.index("--lora")
        if lora_idx + 1 < len(sys.argv) and not sys.argv[lora_idx + 1].startswith("--"):
            ADAPTER_DIR = sys.argv[lora_idx + 1]
        else:
            print("Error: --lora requires an adapter directory path.")
            sys.exit(1)

    TRUST_REMOTE_CODE = "--trust-remote-code" in sys.argv
    num_threads = 0
    if "--threads" in sys.argv:
        idx = sys.argv.index("--threads")
        if idx + 1 < len(sys.argv):
            num_threads = int(sys.argv[idx + 1])
    config_path = os.path.join(MODEL_PATH, "config.json")

    if ADAPTER_DIR:
        trillim_cfg_path = os.path.join(ADAPTER_DIR, "trillim_config.json")
        if not os.path.exists(trillim_cfg_path):
            print(
                f"Error: {trillim_cfg_path} not found.\n"
                "This adapter has not been quantized for Trillim.\n"
                "Run: trillim quantize <model_dir> --adapter "
                f"{ADAPTER_DIR}"
            )
            sys.exit(1)
        lora_path = os.path.join(ADAPTER_DIR, "qmodel.lora")
        if not os.path.exists(lora_path):
            print(
                f"Error: --lora set but {lora_path} not found. "
                "Run: trillim quantize <model_dir> --adapter "
                f"{ADAPTER_DIR}"
            )
            sys.exit(1)

    try:
        tokenizer = load_tokenizer(MODEL_PATH, adapter_dir=ADAPTER_DIR, trust_remote_code=TRUST_REMOTE_CODE)

        arch_config = ArchConfig.from_config_json(config_path, MODEL_PATH, adapter_dir=ADAPTER_DIR)

        from trillim._bin_path import inference_bin

        cmd = [inference_bin(), MODEL_PATH]
        model = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
        )
        engine_options = load_engine_options(num_threads=num_threads)
        model.stdin.write(_build_init_config(arch_config, adapter_dir=ADAPTER_DIR, **engine_options))
        model.stdin.flush()

        sampling_params = load_default_params()

        try:
            _run_chat_loop(model, tokenizer, arch_config, sampling_params)
        finally:
            if model.returncode is None:
                model.terminate()
                try:
                    model.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    model.kill()
                    model.wait()

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


def _run_chat_loop(model, tokenizer, arch_config, sampling_params):
    """Run the interactive chat loop against a running inference subprocess."""
    stop_tokens = set(arch_config.eos_tokens)
    max_context = arch_config.max_position_embeddings
    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template
    )
    messages = []
    cached_token_ids = []
    cached_prompt_str = ""

    model_name = os.path.basename(os.path.normpath(model.args[1]))
    print(f"Talk to {model_name} (Ctrl+D or 'q' to quit, '/new' for new conversation)")
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
                suffix_str = new_prompt[len(cached_prompt_str):]
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

        # Send count-prefixed key=value request block
        model.stdin.write(_build_request_block(delta_tokens, reset_flag, **sampling_params))
        model.stdin.flush()

        print("Model Response: ", end="", flush=True)
        decoder = IncrementalDecoder(tokenizer)
        generated_tokens = []
        response_text = ""
        while True:
            out = _readline_with_timeout(model.stdout, _ENGINE_TIMEOUT)
            if out is None:
                raise RuntimeError(
                    f"Inference engine timed out after {_ENGINE_TIMEOUT}s"
                )
            if not out:
                break
            try:
                token = int(out.strip())
            except ValueError:
                raise RuntimeError(
                    f"Protocol error — expected int token_id, got {out.strip()!r}"
                )
            if token in stop_tokens:
                generated_tokens.append(token)
                break

            generated_tokens.append(token)
            new_text = decoder.decode(token)
            response_text += new_text
            print(new_text, end="", flush=True)
        print()

        # Read kv_position line
        kv_line = _readline_with_timeout(model.stdout, _ENGINE_TIMEOUT)
        if kv_line is None:
            raise RuntimeError(
                f"Inference engine timed out after {_ENGINE_TIMEOUT}s"
            )
        if kv_line:
            try:
                kv_position = int(kv_line.strip())
            except ValueError:
                raise RuntimeError(
                    f"Protocol error — expected int kv_position, got {kv_line.strip()!r}"
                )
            cached_token_ids = (all_token_ids + generated_tokens)[:kv_position]

        # Update cached prompt string and message history
        messages.append({"role": "assistant", "content": response_text})
        if has_chat_template:
            cached_prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )


if __name__ == "__main__":
    main()
