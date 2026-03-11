# Python Components

Use these classes when you want to embed Trillim directly in Python instead of calling the CLI.

`Runtime(...)` is the recommended entry point for synchronous applications. It owns the event loop, starts components in order, stops them in reverse order, and exposes component methods synchronously as `runtime.llm`, `runtime.whisper`, and `runtime.tts`.

## Before You Start

- Any `LLM(...)` example on this page assumes `trillim pull Trillim/BitNet-TRNQ`
- Any `Whisper(...)` or `TTS()` example requires `uv add "trillim[voice]"` or `pip install "trillim[voice]"`
- `Whisper(model_size="base.en")` downloads its checkpoint on first start
- `TTS()` works with built-in voices immediately and can also persist custom voices

## Use `Runtime`

```python
from trillim import LLM, Runtime, TTS, Whisper

runtime = Runtime(
    LLM("~/.trillim/models/Trillim/BitNet-TRNQ"),
    Whisper(model_size="base.en"),
    TTS(),
)

runtime.start()
try:
    messages = [{"role": "user", "content": "Write a one-line haiku about CPUs."}]
    reply = runtime.llm.chat(messages, timeout=30)
    print(reply)

    for event in runtime.llm.stream_chat(messages):
        if event.type == "token":
            print(event.text, end="", flush=True)
    text = runtime.whisper.transcribe_wav("recording.wav", timeout=30)
finally:
    runtime.stop()
```

Use `with Runtime(...) as runtime:` when you want automatic teardown.

## Component Lifecycle

Use direct async lifecycle management when you want full control over the event loop yourself.

- Import the component you need from `trillim`
- Call `await component.start()` before using public component methods or advanced internals like `component.engine` and `llm.harness`
- Call `await component.stop()` when finished, ideally in a `finally` block
- Use `router()` only when mounting the component into your own FastAPI app

## Use `LLM` Directly

```python
import asyncio

from trillim import LLM


async def main():
    llm = LLM("~/.trillim/models/Trillim/BitNet-TRNQ")
    await llm.start()
    try:
        messages = [{"role": "user", "content": "Write a one-line haiku about CPUs."}]
        reply = await llm.chat(messages, timeout=30)
        print(reply)
    finally:
        await llm.stop()


asyncio.run(main())
```

After `await llm.start()`, use `llm.count_tokens(messages)`, `llm.max_context_tokens`, and `llm.validate_context(messages)` to inspect prompt size safely from app code.

```python
from trillim import ContextOverflowError

messages = [{"role": "user", "content": "Write a one-line haiku about CPUs."}]

try:
    prompt_tokens = llm.validate_context(messages)
    print(f"{prompt_tokens=}, {llm.max_context_tokens=}")
except ContextOverflowError as exc:
    print(exc)
```

Use `llm.stream_chat(...)` when you want structured progress events:

```python
async for event in llm.stream_chat(messages):
    if event.type == "search_started":
        print(f"Searching for: {event.query}")
    elif event.type == "search_result":
        print(f"Search available: {event.available}")
    elif event.type == "token":
        print(event.text, end="", flush=True)
    elif event.type == "final_text":
        print(f"\nFinal: {event.text}")
```

`llm.harness.run(...)` remains available as an advanced text-only interface.

Use `llm.engine` only when you need lower-level control, such as direct token generation, custom prompt assembly, or explicit tokenizer access.

## Use `Whisper` and `TTS` Directly

```python
import asyncio
from pathlib import Path

from trillim import TTS, Whisper


async def main():
    whisper = Whisper(model_size="base.en")
    tts = TTS()
    await whisper.start()
    await tts.start()
    try:
        text = await whisper.transcribe_wav(Path("recording.wav"), timeout=30)
        audio = await tts.engine.synthesize_full(text, voice="alba")
        Path("speech.wav").write_bytes(audio)
    finally:
        await whisper.stop()
        await tts.stop()


asyncio.run(main())
```

## Compose a Server in Python

```python
from trillim import Server, LLM, Whisper, TTS

# Inference only
Server(LLM("~/.trillim/models/Trillim/BitNet-TRNQ")).run()

# Inference + voice pipeline
Server(
    LLM("~/.trillim/models/Trillim/BitNet-TRNQ"),
    Whisper(model_size="base.en"),
    TTS(),
).run()

# TTS only
Server(TTS()).run()
```

## Constructor Reference

### `LLM`

```python
from trillim import LLM

LLM(
    model_dir="~/.trillim/models/Trillim/BitNet-TRNQ",
    adapter_dir=None,          # optional LoRA adapter path
    num_threads=0,             # 0 = auto-detect
    trust_remote_code=False,
    lora_quant=None,           # "none", "bf16", "int8", "q4_0", etc.
    unembed_quant=None,        # "int8", "q4_0", etc.
    harness_name="default",    # "default" or "search"
)
```

`harness_name="search"` uses `ddgs` by default. If you switch a running server to Brave search later, set `SEARCH_API_KEY` and call `POST /v1/models/load` with `search_provider`.

## `Whisper`

```python
from trillim import Whisper

Whisper(
    model_size="base.en",
    compute_type="int8",
    cpu_threads=2,
)
```

After `await whisper.start()`, prefer the public helpers below. `whisper.engine.transcribe(...)` remains available as an advanced escape hatch.

Preferred public helpers:

```python
text = await whisper.transcribe_bytes(wav_bytes, language="en", timeout=30)
text = await whisper.transcribe_wav("recording.wav", timeout=30)
text = await whisper.transcribe_array(samples, sample_rate=44100, timeout=30)
```

## `TTS`

```python
from trillim import TTS

TTS(
    voices_dir="~/.trillim/voices",
)
```

After `await tts.start()`, use `tts.engine.list_voices()`, `await tts.engine.register_voice(...)`, `await tts.engine.synthesize_full(...)`, or `tts.engine.synthesize_stream(...)`.

## Add Custom Routes

```python
from trillim import Server, LLM

server = Server(LLM("~/.trillim/models/Trillim/BitNet-TRNQ"))
app = server.app

@app.get("/health")
async def health():
    return {"status": "ok"}

server.run(host="0.0.0.0", port=8000)
```
