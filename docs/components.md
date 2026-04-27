# Python SDK

Use the Python SDK when you want to embed Trillim in an application instead of shelling out to the CLI.

If you want the deeper operational details behind the public surface, see [Advanced SDK and Server Notes](advanced.md).

## Choose Sync or Async

Trillim exposes both patterns:

- `Runtime(...)` is the synchronous facade. It starts components on a background event loop and lets you call them from normal blocking Python.
- `LLM`, `STT`, and `TTS` can also be used directly as async components.

## Use `Runtime` for Synchronous Code

`Runtime` is the easiest entry point for scripts, workers, and sync web backends.

```python
from trillim import LLM, Runtime

with Runtime(LLM("Trillim/BitNet-TRNQ")) as runtime:
    with runtime.llm.open_session() as session:
        reply = session.collect("Explain local inference in one sentence.")
        print(reply)
```

`Runtime` exposes composed components by name:

- `runtime.llm`
- `runtime.stt`
- `runtime.tts`

It also syncifies async iterators and async context-managed session objects, so these patterns are valid:

```python
from trillim import LLM, Runtime
from trillim.components.llm import ChatTokenEvent, ChatDoneEvent

with Runtime(LLM("Trillim/BitNet-TRNQ")) as runtime:
    with runtime.llm.open_session() as session:
        for event in session.generate("Write five words about CPUs."):
            if isinstance(event, ChatTokenEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, ChatDoneEvent):
                print(f"\nused {event.usage.total_tokens} tokens")
```

## Use `LLM` Directly for Async Code

```python
import asyncio

from trillim import LLM


async def main():
    llm = LLM("Trillim/BitNet-TRNQ")
    await llm.start()
    try:
        async with llm.open_session() as session:
            reply = await session.collect("Name two benefits of local models.")
            print(reply)
    finally:
        await llm.stop()


asyncio.run(main())
```

### `LLM` Constructor Rules

`LLM(...)` loads from a managed store ID, not a raw path:

```python
from trillim import LLM

llm = LLM("Trillim/BitNet-TRNQ")
adapter_llm = LLM(
    "Trillim/BitNet-TRNQ",
    lora_dir="Trillim/BitNet-GenZ-LoRA-TRNQ",
)
```

Useful constructor options:

| Option | Meaning |
| --- | --- |
| `num_threads` | Worker thread count, `0` uses Trillim defaults |
| `lora_dir` | Optional adapter store ID |
| `lora_quant` | Runtime quantization for LoRA layers |
| `unembed_quant` | Runtime quantization for the unembedding layer |
| `trust_remote_code` | Opt in to custom bundle tokenizer/config code |
| `harness_name` | `default` or `search` |
| `search_provider` | `ddgs` or `brave` |
| `search_token_budget` | Search-context budget; clamped at runtime to one quarter of the active model context window |

### One-Turn Calls and Sessions

The current SDK exposes LLM generation through `ChatSession`. For a one-turn call, open a session and call `collect(user_message)`. For streaming, iterate `generate(user_message)`.

Use `append_message(role, content)` to preload system, user, assistant, or search context before the next generated user turn.

Use the same session when you want multi-turn state:

```python
import asyncio

from trillim import LLM


async def main():
    llm = LLM("Trillim/BitNet-TRNQ")
    await llm.start()
    try:
        async with llm.open_session() as session:
            session.append_message("system", "Be concise.")
            print(await session.collect("Give me three uses for local AI."))
            print(await session.collect("Now shorten that to one sentence."))
    finally:
        await llm.stop()


asyncio.run(main())
```

Session rules that matter in real code:

- `ChatSession` is created by `LLM.open_session()`. You do not construct it yourself.
- `open_session()` does not take initial messages. Add existing context with `append_message()`.
- `collect(user_message)` returns the final assistant string.
- `generate(user_message)` yields `ChatTokenEvent`, `ChatFinalTextEvent`, and `ChatDoneEvent`.
- `new_chat()` clears committed conversation history on an idle session.
- A session is single-consumer. Do not iterate and mutate it concurrently.
- When a model swap begins, existing chat sessions become stale and raise `SessionStaleError`.
- A closed session raises `SessionClosedError` if reused.
- Very long-lived sessions can hit the lifetime token cap and raise `SessionExhaustedError`.

Final practical note: avoid keeping multiple active `ChatSession` objects open from the same `LLM` component. The underlying inference engine only caches the latest chat thread. Alternating between sessions will constantly evict and rebuild that cache, which is slow and can make generation hit progress timeouts on CPU-bound machines. Prefer one live chat session per `LLM`, or run independent conversations through separate `LLM` components when isolation matters.

### Enable Search

The search harness is for models that emit `<search>...</search>` tags.

```python
from trillim import LLM

llm = LLM(
    "Trillim/BitNet-TRNQ",
    lora_dir="Trillim/BitNet-Search-LoRA-TRNQ",
    harness_name="search",
    search_provider="ddgs",
)
```

Notes:

- Use `brave` only if `SEARCH_API_KEY` is set in the environment.
- Search is configured from the SDK or Python server composition, not the CLI.

### Hot-Swap a Running Model

```python
import asyncio

from trillim import LLM


async def main():
    llm = LLM("Trillim/BitNet-TRNQ")
    await llm.start()
    try:
        await llm.swap_model(
            "Trillim/BitNet-TRNQ",
            lora_dir="Trillim/BitNet-GenZ-LoRA-TRNQ",
        )
        async with llm.open_session() as session:
            print(await session.collect("Confirm the adapter is active."))
    finally:
        await llm.stop()


asyncio.run(main())
```

Important hot-swap behavior:

- The component must already be running.
- HTTP swap requests fail fast when the LLM route is already handling work. Direct SDK calls are serialized by the component locks.
- Existing sessions become stale once swap handoff begins.
- The current implementation stops the old engine before starting the replacement engine. If replacement startup fails, the LLM becomes unavailable until restarted or swapped again.
- `stop()` wins over in-flight startup, hot swap, and recovery restart work; if shutdown races with replacement-model preflight or handoff, Trillim discards that work and leaves the component `unavailable`.
- Omitted init-time options reset to Trillim defaults instead of inheriting the previous runtime.

## Use `STT`

`STT` requires the `voice` extra.

```python
import asyncio
from pathlib import Path

from trillim import STT


async def main():
    stt = STT()
    await stt.start()
    try:
        async with stt.open_session() as session:
            print(await session.transcribe(Path("sample.wav").read_bytes(), language="en"))
    finally:
        await stt.stop()


asyncio.run(main())
```

Public helpers:

- `stt.open_session()`
- `await session.transcribe(audio_bytes, language=None)`

Practical notes:

- `STTSession` is created by `STT.open_session()`. You do not construct it directly.
- `session.transcribe()` accepts signed 16-bit little-endian mono `16 kHz` PCM bytes or WAV that Trillim converts to that PCM format.
- `language` is optional and must contain only letters and hyphens.
- `STT` serializes engine use to one transcription at a time. SDK callers queue cooperatively behind that slot instead of failing fast.
- Direct async `STT` use is bound to one event loop. Create a new `STT()` per thread or event loop.

## Use `TTS`

`TTS` also requires the `voice` extra.
`TTS()` does not take constructor options; choose `voice` and `speed` when opening a session.

```python
import asyncio
from pathlib import Path

from trillim import TTS


async def main():
    tts = TTS()
    await tts.start()
    try:
        print(await tts.list_voices())

        async with await tts.open_session(voice="alba", speed=1.0) as session:
            pcm = await session.collect("Hello from Trillim.")
            Path("speech.pcm").write_bytes(pcm)

            async for chunk in session.synthesize("Streaming speech."):
                print(len(chunk))
    finally:
        await tts.stop()


asyncio.run(main())
```

Public helpers:

- `await tts.list_voices()`
- `await tts.register_voice(name, audio)`
- `await tts.delete_voice(name)`
- `await tts.open_session(voice=None, speed=None)`

`audio` for `register_voice()` can be:

- `bytes`
- `str`
- `Path`

Custom voice names and `voice` selectors accept only ASCII letters and digits.
Custom voices are stored as Pocket TTS-native `.safetensors` under `~/.trillim/voices`.
Legacy `.state` files and invalid safetensors files are skipped at startup with warnings and do not appear in `list_voices()`.
The runtime voice cache is the source of truth while the component is running; disk storage is kept in sync on a best-effort basis.
To replace a custom voice, delete it first and then register the same name again. Built-in voices cannot be deleted or shadowed.

Session rules that matter:

- `TTSSession` is created by `TTS.open_session()`. You do not construct it directly.
- A session is reusable. Call `collect(text)` or iterate `synthesize(text)` for each synthesis turn.
- `collect(text)` returns raw PCM bytes: `24 kHz`, mono, signed `16-bit` little-endian.
- `synthesize(text)` yields raw PCM chunks with the same format.
- A session is single-consumer while one synthesis is active; starting a second synthesis on the same session raises `SessionBusyError`.
- `set_voice()` is allowed only when the session is idle or done.
- `set_speed()` is allowed during synthesis and is a best-effort live update for later segments.
- `pause()` stops yielding queued chunks to the caller until `resume()`.
- `close()` cancels any active synthesis, clears buffered audio, and leaves the session reusable.
- If a session outlives `TTS.stop()`, the next synthesis raises `ComponentLifecycleError`; stopped sessions do not return successful empty audio.
- Direct async `TTS` use is bound to one event loop. Create a new `TTS()` per thread or event loop.
- The SDK serializes engine access internally. The HTTP router enforces one live TTS request at a time across speech and voice-management routes and rejects concurrent requests with `429`.

## Public Error Types You Will See First

- `InvalidRequestError`: input validation failed before work started
- `AdmissionRejectedError`: the component is busy or draining
- `ContextOverflowError`: the rendered prompt exceeded the active model context window
- `ProgressTimeoutError`: an operation stopped making required progress
- `SessionBusyError`: a session already has an active consumer
- `SessionClosedError`: a closed session was reused
- `SessionStaleError`: an LLM session was invalidated by model swap
- `SessionExhaustedError`: an LLM session exceeded its lifetime token quota

## When to Switch to the Server API

Use the SDK when you want direct control from Python. Switch to the [server docs](server.md) when you need:

- HTTP access from another process or machine
- OpenAI client compatibility
- a health endpoint
- raw-body voice routes
- more operational detail in one place via [Advanced SDK and Server Notes](advanced.md)
