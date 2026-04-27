# Advanced SDK and Server Notes

This page collects the deeper operational details behind Trillim's SDK and server surfaces. It is for users who want to make the most of the runtime.

## Runtime Model

`Runtime` is the supported synchronous facade over async-native components.

What that means in practice:

- `LLM`, `STT`, and `TTS` are async components internally
- `Runtime` owns a background event loop and exposes blocking sync calls
- component methods, async iterators, and session handles are all bridged through that runtime loop

This is why sync control operations still behave cooperatively rather than magically interrupting work mid-step.

## Runtime Metadata

The current public SDK keeps LLM runtime introspection small. `LLM.open_session()` is the main readiness check for SDK callers: it succeeds only while the component is running and raises `ComponentLifecycleError` otherwise.

The HTTP server exposes the active model ID through `GET /v1/models`. It does not currently return adapter paths, context-window size, trust settings, or worker init options.

## Session Handles

`ChatSession`, `STTSession`, and `TTSSession` are live handles, not plain value objects.

Important rules:

- sessions are created by their owning service and are never caller-constructed
- the public session types are abstract handles; concrete implementations stay private
- sessions are single-consumer
- direct async component and session access is bound to the owning event loop
- explicit close behavior is part of the contract
- `Runtime` is the supported sync context-manager surface for returned session handles

### `ChatSession` States

- `idle`
- `streaming`
- `done`
- `stale`

Important implications:

- a session becomes `stale` when an LLM swap actually begins handoff after preflight succeeds
- `close()` cancels an active turn, waits for cleanup, and returns the session to `idle`
- a session becomes `done` after caller-side generator cancellation, hard engine failure, or lifetime-token exhaustion

### `STTSession` States

- `idle`
- `transcribing`
- `done`

Important implications:

- `transcribe(audio, language=None)` starts one transcription turn
- a session is single-use after transcription starts; it moves to `done` when the turn finishes or fails
- `close()` is currently a no-op; STT bulk transcription does not expose cancellable caller-visible streaming state
- stopped owner components surface as `ComponentLifecycleError`

### `TTSSession` States

- `idle`
- `running`
- `paused`
- `done`

Important implications:

- `collect(text)` and `synthesize(text)` start one synthesis turn on a reusable session
- only one synthesis turn can be active per session
- `pause()` is consumer-visible: queued chunks are not yielded until `resume()`
- the producer may continue synthesizing ahead until the bounded queue fills
- `set_voice()` is rejected while synthesis is active
- `set_speed()` is best-effort during active synthesis and affects later post-processing
- `close()` cancels any active synthesis, clears buffered audio, and returns the session to `idle`
- stopped owner components surface as `ComponentLifecycleError`; sessions do not silently complete as successful empty audio after `TTS.stop()`

## Safe-Boundary Control Semantics

Trillim uses cooperative control semantics rather than pretending work can be interrupted anywhere.

Safe boundaries:

- `LLM`: emitted token or stream chunk boundaries
- `STT`: no caller-visible incremental control surface; work starts only after the full bounded upload is accepted
- `TTS`: text-segment, emitted-audio, and bounded-queue boundaries

This matters for close/cancellation, pause, resume, and speed changes.

## Search Harness Contract

The search harness is not just “LLM plus search results.” It has its own behavior model.

Rules:

- it looks for model-emitted `<search>...</search>` sentinels
- search-detection turns are buffered until the harness knows whether a search sentinel was emitted
- if search is used, only the final answer turn is streamed incrementally to the caller
- if no search sentinel is emitted, the buffered answer is emitted only after that turn completes
- search history may appear in session state as role `search`
- before prompt rendering, `search` history entries are normalized into `system` messages prefixed with `Search results:`

Operational limits:

- total model turns per request: `3`
- max search fetch rounds per request: `2`
- supported providers: `ddgs`, `brave`
- results per search: `5`
- runtime search token budget is clamped to one quarter of the active model context window

`brave` requires `SEARCH_API_KEY` in the environment.

## Hot Swap Semantics

Hot swap is designed to preserve correctness first.

Important rules:

- swap can change the base model, adapter, and search/runtime options together
- HTTP swap requests fail fast when the LLM route is already handling work; direct SDK swaps are serialized by component locks
- the current implementation stops the old engine before starting the replacement engine to avoid overlapping model memory
- once swap handoff begins, existing chat sessions become stale
- `stop()` is authoritative across startup, hot swap, and recovery restart; if stop wins a race, any preflighted or partially started replacement runtime is discarded and the component remains `unavailable`
- omitted or `null` init-time swap fields reset to Trillim defaults rather than inheriting the previous runtime's values

If you expose `/v1/models/swap`, design your callers around session invalidation and verify the active model ID with `/v1/models` after a successful swap.

## Adapter Overlay Rules

Important rules:

- the base `model_dir` remains the source of required runtime artifacts such as weights and rope cache
- adapter compatibility is validated through required `base_model_config_hash` metadata
- adapter tokenizer and config metadata take precedence when an adapter is configured; base-model metadata is the fallback for missing values
- runtime overlays are manifest-driven and materialize only the files Trillim actually needs
- `qmodel.tensors`, `rope.cache`, and `qmodel.lora` are hardlinked into the overlay
- if a hardlink cannot be created, startup or swap fails instead of copying those artifacts
- `model_dir`, `lora_dir`, and the process temp directory must share a filesystem for LoRA overlays to work

The practical takeaway is simple: keep the base model, adapter, and temp area on the same filesystem and treat adapter compatibility metadata as required, not advisory.

## `trust_remote_code` Boundaries

`trust_remote_code` is intentionally fail-closed.

Rules:

- Trillim will not load bundles that require remote code unless you opt in
- when enabled with adapters, adapter files may override tokenizer or config Python modules in the runtime overlay
- only top-level local `auto_map` entry points and their bounded sibling-module relative import closure are supported
- package-scoped `auto_map` modules and package or parent relative imports are not supported

The built-in CLI defaults to `trust_remote_code=False`.

## Limits and Capacity

The runtime avoids unbounded queues and unbounded retries by design.

### LLM

- request body cap: `2 MiB`
- messages per request: `256`
- pre-tokenization text cap per request: `1 MiB`
- lifetime quota per chat session: `256k` tokens
- max generated output per request: `8192` tokens
- active generations: `1`
- queue length: `0`

### STT

- upload cap: `64 MiB`
- SDK engine concurrency: `1`
- SDK queueing: unbounded cooperative wait behind the single engine slot
- HTTP active jobs: `1`
- HTTP queue length: `0`
- transcription worker timeout: `180s`

### TTS

- input text cap: `1_000_000` chars
- raw HTTP speech body cap: `6 MiB`
- HTTP active jobs: `1`
- HTTP queue length: `0`
- SDK engine concurrency: `1`
- hard text-segment cap: `512` chars
- emitted audio chunk queue: `8`

### Custom Voices

- max stored voices: `64`
- built-in Pocket TTS voices do not count against the custom voice quota
- max upload size per voice: `10 MiB`
- max serialized voice state per voice: `64 MiB`
- total stored custom voice bytes: `100 MiB`
- voice-state build timeout: `30s`
- custom voice files use Pocket TTS-native `.safetensors`
- legacy `.state` files and invalid voice payloads are skipped with warnings at startup
- runtime voice state is authoritative while `TTS` is running; on-disk custom voice storage is synchronized best-effort
- custom voices can be replaced by deleting and then registering the same name; built-in voices cannot be deleted or shadowed

## Progress Timeout Model

Timeouts are progress-based by default.

Rules:

- `LLM`: the next token or output chunk must arrive within `5s`
- `STT`: once transcription begins, the worker must complete within `180s`
- `TTS`: the next emitted audio chunk must arrive within `60s`
- custom voice registration must complete voice-state build within `30s`
- non-progress heartbeats do not count
- if progress stops, the worker is killed and the request fails

For STT specifically:

- the HTTP ingress path enforces `content-type`, `content-length`, upload progress timeout, and total upload timeout before transcription begins
- the SDK path is intentionally lighter and relies on caller discipline rather than the full HTTP hardening boundary

For TTS specifically:

- the HTTP TTS routes enforce single-request admission across speech and voice-management requests and reject concurrent requests with `429`
- the SDK path allows multiple sessions, but all engine calls are serialized by the owning `TTS` instance
- direct async `TTS` and `TTSSession` use is bound to one event loop; create one component instance per thread or event loop

This is why a timeout in Trillim often implies worker recovery, not just a raised error.

## Choosing the Right Surface

Use the main docs first:

- [Python SDK](components.md) for normal integration work
- [API Server](server.md) for routes and client-facing behavior

Use this page when you need to reason about:

- exact session invalidation and cleanup behavior
- hot-swap edge cases
- search orchestration behavior
- LoRA overlay constraints
- timeout, quota, and admission behavior
