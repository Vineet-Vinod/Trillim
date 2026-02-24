# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
import os
import subprocess
import sys
import tempfile
import threading

from prompt_toolkit import prompt as better_input
from prompt_toolkit.key_binding import KeyBindings
from trillim.model_arch import ModelConfig as ArchConfig
from trillim.token_utils import IncrementalDecoder
from trillim.utils import (
    load_tokenizer,
    load_default_params,
    load_engine_options,
    _build_init_config,
    _build_request_block,
)

_ENGINE_TIMEOUT = 300  # seconds; maximum wait for a single engine I/O operation


def _make_key_bindings():
    """Create key bindings for the chat prompt. Ctrl+G opens $EDITOR."""
    kb = KeyBindings()

    @kb.add("c-g")
    def _open_editor(event):
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR", "vi")
        buf = event.app.current_buffer
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
            f.write(buf.text)
            tmp_path = f.name
        try:
            subprocess.call([editor, tmp_path])
            with open(tmp_path) as f:
                text = f.read()
            buf.text = text
            buf.cursor_position = len(text)
        finally:
            os.unlink(tmp_path)

    return kb


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


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: trillim chat <model_directory> [--lora <adapter_dir>] "
            "[--threads N] [--lora-quant TYPE] [--unembed-quant TYPE]"
        )

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
            raise ValueError("--lora requires an adapter directory path.")

    TRUST_REMOTE_CODE = "--trust-remote-code" in sys.argv
    num_threads = 0
    if "--threads" in sys.argv:
        idx = sys.argv.index("--threads")
        if idx + 1 < len(sys.argv):
            num_threads = int(sys.argv[idx + 1])
    lora_quant = None
    if "--lora-quant" in sys.argv:
        idx = sys.argv.index("--lora-quant")
        if idx + 1 < len(sys.argv):
            lora_quant = sys.argv[idx + 1]
    unembed_quant = None
    if "--unembed-quant" in sys.argv:
        idx = sys.argv.index("--unembed-quant")
        if idx + 1 < len(sys.argv):
            unembed_quant = sys.argv[idx + 1]
    config_path = os.path.join(MODEL_PATH, "config.json")

    if ADAPTER_DIR:
        trillim_cfg_path = os.path.join(ADAPTER_DIR, "trillim_config.json")
        if not os.path.exists(trillim_cfg_path):
            raise FileNotFoundError(
                f"{trillim_cfg_path} not found. "
                "This adapter has not been quantized for Trillim. "
                f"Run: trillim quantize <model_dir> --adapter {ADAPTER_DIR}"
            )
        lora_path = os.path.join(ADAPTER_DIR, "qmodel.lora")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"--lora set but {lora_path} not found. "
                f"Run: trillim quantize <model_dir> --adapter {ADAPTER_DIR}"
            )
        from trillim.model_store import validate_adapter_model_compat
        validate_adapter_model_compat(ADAPTER_DIR, MODEL_PATH)

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
        engine_options = load_engine_options(num_threads=num_threads, lora_quant=lora_quant, unembed_quant=unembed_quant)
        model.stdin.write(_build_init_config(arch_config, adapter_dir=ADAPTER_DIR, **engine_options))
        model.stdin.flush()

        sampling_params = load_default_params(MODEL_PATH)

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
    kb = _make_key_bindings()
    print(f"Talk to {model_name} (Ctrl+D or 'q' to quit, '/new' for new conversation, Ctrl+G for editor)")
    while True:
        try:
            query = better_input("> ", key_bindings=kb)
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
