# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
import asyncio
import os
import subprocess
import sys
import tempfile

from prompt_toolkit import prompt as better_input
from prompt_toolkit.key_binding import KeyBindings

from trillim.errors import ContextOverflowError

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


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: trillim chat <model_directory> [--lora <adapter_dir>] "
            "[--threads N] [--lora-quant TYPE] [--unembed-quant TYPE] "
            "[--harness NAME] [--search-provider NAME]"
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
    harness_name = "default"
    if "--harness" in sys.argv:
        idx = sys.argv.index("--harness")
        if idx + 1 < len(sys.argv):
            harness_name = sys.argv[idx + 1]
    search_provider = "ddgs"
    if "--search-provider" in sys.argv:
        idx = sys.argv.index("--search-provider")
        if idx + 1 < len(sys.argv):
            search_provider = sys.argv[idx + 1]

    try:
        from trillim.server import LLM

        llm = LLM(
            MODEL_PATH,
            adapter_dir=ADAPTER_DIR,
            num_threads=num_threads,
            trust_remote_code=TRUST_REMOTE_CODE,
            lora_quant=lora_quant,
            unembed_quant=unembed_quant,
            harness_name=harness_name,
        )
        llm._search_provider = search_provider

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(llm.start())
            try:
                assert llm.engine is not None
                _run_chat_loop(loop, llm, llm.engine.default_params)
            finally:
                loop.run_until_complete(llm.stop())
        finally:
            loop.close()

    except BrokenPipeError:
        print("\nError: Inference engine crashed.")
        print("\nIf you think the engine is broken, please report the bug!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


def _run_chat_loop(loop, llm, sampling_params):
    """Interactive chat loop — sync input, async generation via ChatSession."""
    model_name = llm.model_name
    max_context = llm.max_context_tokens
    chat = llm.session()
    chat_sampling = {
        key: sampling_params[key]
        for key in (
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "max_tokens",
        )
        if key in sampling_params
    }

    kb = _make_key_bindings()
    print(f"Talk to {model_name} (Ctrl+D or 'q' to quit, '/new' for new conversation, Ctrl+G for editor)")
    while True:
        try:
            query = better_input("> ", key_bindings=kb)
        except (EOFError, KeyboardInterrupt):
            query = "q"

        if query.strip() == "q":
            break

        if query.strip() == "/new":
            chat = llm.session()
            assert llm.engine is not None
            llm.engine.reset_prompt_cache()
            print("Starting new conversation.")
            continue

        chat.add_user(query)
        try:
            chat.validate()
        except ContextOverflowError:
            print(
                f"Context window full ({max_context} tokens). Starting new conversation."
            )
            latest_message = chat.messages[-1]
            assert llm.engine is not None
            llm.engine.reset_prompt_cache()
            chat = llm.session([latest_message])
            try:
                chat.validate()
            except ContextOverflowError:
                print(
                    f"Last message exceeds the context window ({max_context} tokens). Shorten it and try again."
                )
                chat = llm.session()
                continue

        loop.run_until_complete(_stream_response(chat, chat_sampling))
        print()


async def _stream_response(chat, sampling_params):
    """Drain chat-session events, printing status and text as they arrive."""
    async for event in chat.stream_chat(**sampling_params):
        if event.type == "search_started":
            print(f"[Searching: {event.query}]", flush=True)
        elif event.type == "search_result" and not event.available:
            print("[Search unavailable]", flush=True)
        elif event.type == "search_result":
            print("[Synthesizing...]", flush=True)
        elif event.type == "token":
            print(event.text, end="", flush=True)


if __name__ == "__main__":
    main()
