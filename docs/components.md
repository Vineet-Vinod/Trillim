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
    session = runtime.tts.speak("Hello there", speed=1.25, interrupt=True, timeout=30)
    session.set_speed(1.5)
    for chunk in session:
        print(len(chunk))
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
        session = tts.speak(text, voice="alba", speed=1.25, timeout=30)
        session.set_speed(1.5)
        audio = await session.collect()
        Path("speech.pcm").write_bytes(audio)
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
    default_voice="alba",
    speed=1.0,
)
```

After `await tts.start()`, use the public component methods:

```python
voices = tts.list_voices()
tts.default_voice = "jean"
tts.speed = 1.5
sample_rate = tts.sample_rate

await tts.register_voice("myvoice", wav_bytes)
pcm_chunks = [chunk async for chunk in tts.synthesize_stream("Hello there", speed=1.25)]
wav_bytes = await tts.synthesize_wav("Hello there", voice="myvoice", speed=1.5)
session = tts.speak("Queued speech", interrupt=False, timeout=30)
session.set_speed(1.25)
session.pause()
session.resume()
audio = await session.collect()
await tts.delete_voice("myvoice")
```

`tts.engine` remains available as an advanced escape hatch.
`speed` accepts values from `0.25` to `4.0` and uses pitch-preserving time stretching rather than naive resampling.
Speed-adjusted synthesis streams progressively with bounded lookahead; it does not wait for the full utterance before yielding audio.
`tts.speak(...)` returns a `TTSSession` that queues behind the active session by default. Pass `interrupt=True` to cancel the active and queued sessions before starting the new one.
`TTSSession` yields PCM chunks at `tts.sample_rate`.
`pause()` and `resume()` control future chunk production only. They do not control speaker-device playback.
`set_speed()` changes future chunk production only. Already emitted audio is unchanged, and already buffered audio may still arrive at the old speed for a short bounded window.
For `Runtime`, dynamic speed control is effective when you consume a session progressively. If you call `session.collect()` and wait for the full utterance first, there is no opportunity to adjust speed mid-stream.

## `SentenceChunker`

```python
from trillim import SentenceChunker

chunker = SentenceChunker()
for piece in chunker.feed("Hello world. Another sentence"):
    print(piece)
print(chunker.flush())
```

Use `SentenceChunker` to split streaming LLM output into sentence-sized chunks before handing it to TTS.

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
