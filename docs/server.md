# API Server

Trillim exposes a small FastAPI server for local HTTP access.

If you want the deeper operational semantics behind the server surface, see [Advanced SDK and Server Notes](advanced.md).

There are two ways to run it:

- `trillim serve Trillim/<name>` for the built-in demo server
- `Server(...)` in Python when you need more control

## CLI Server vs Python Server

The distinction matters:

| Need | `trillim serve` | `Server(...)` |
| --- | --- | --- |
| quick local server | yes | yes |
| fixed `127.0.0.1:8000` bind | yes | configurable |
| custom host/port | no | yes |
| `/v1/models/swap` | no | yes, with `allow_hot_swap=True` |
| custom search setup | no | yes |

### CLI

```bash
trillim serve Trillim/BitNet-TRNQ
```

With voice routes:

```bash
trillim serve Trillim/BitNet-TRNQ --voice
```

### Python

```python
from trillim import LLM, Server

server = Server(
    LLM("Trillim/BitNet-TRNQ"),
    allow_hot_swap=True,
)
server.run(host="127.0.0.1", port=8000)
```

## Endpoints

| Route | Method | Purpose |
| --- | --- | --- |
| `/healthz` | `GET` | readiness and component health |
| `/v1/models` | `GET` | active model metadata |
| `/v1/chat/completions` | `POST` | OpenAI-compatible chat completions |
| `/v1/models/swap` | `POST` | optional hot-swap route |
| `/v1/audio/transcriptions` | `POST` | optional STT route |
| `/v1/audio/speech` | `POST` | optional TTS route |
| `/v1/voices` | `GET` | optional voice list |
| `/v1/voices` | `POST` | optional custom voice upload |
| `/v1/voices/{voice_name}` | `DELETE` | optional custom voice deletion |

There is no `/v1/completions` route in this implementation.

## `GET /healthz`

Returns `200` when all composed components are healthy:

```json
{"status": "ok"}
```

If an LLM component is not in the `running` state, the server returns `503` and includes the component state:

```json
{
  "status": "degraded",
  "components": {
    "llm": {
      "state": "swapping"
    }
  }
}
```

## `GET /v1/models`

Returns truthful metadata for the active runtime:

```json
{
  "object": "list",
  "state": "running",
  "data": [
    {
      "id": "BitNet-TRNQ",
      "object": "model",
      "path": "/Users/you/.trillim/models/Trillim/BitNet-TRNQ",
      "max_context_tokens": 4096,
      "trust_remote_code": false,
      "adapter_path": null,
      "init_config": {
        "num_threads": 0,
        "lora_quant": null,
        "unembed_quant": null
      }
    }
  ]
}
```

## `POST /v1/chat/completions`

This is the OpenAI-compatible chat route.

Minimal request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Give me one sentence about local inference."}
    ]
  }'
```

Minimal Python client example:

```python
import json
import urllib.request

body = json.dumps(
    {
        "model": "BitNet-TRNQ",
        "messages": [{"role": "user", "content": "Say hello."}],
    }
).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=60) as response:
    payload = json.loads(response.read().decode("utf-8"))

print(payload["choices"][0]["message"]["content"])
```

Supported request fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `messages` | array | required message list |
| `model` | string | optional, but if present it must match the active model name |
| `stream` | bool | enable SSE streaming |
| `temperature` | float | `0.0` to `2.0` |
| `top_k` | int | `1` to `200` |
| `top_p` | float | `> 0.0` and `<= 1.0` |
| `repetition_penalty` | float | `> 0.0` and `<= 2.0` |
| `rep_penalty_lookback` | int | `>= 0` |
| `max_tokens` | int | `0` for unlimited, or `1` to `8192` |

Notes:

- Typical clients should send only `system`, `user`, and `assistant` roles.
- When the LLM is using the search harness, the OpenAI route still streams assistant text only. Internal search progress is not exposed on this endpoint.
- Requests larger than the JSON body cap are rejected before processing.

### Streaming

Set `"stream": true` to receive server-sent events:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hi."}],
    "stream": true
  }'
```

The stream follows OpenAI-style chat chunks and ends with:

```text
data: [DONE]
```

## `POST /v1/models/swap`

This route exists only when the server was created with `allow_hot_swap=True`.

Example:

```bash
curl http://127.0.0.1:8000/v1/models/swap \
  -H "content-type: application/json" \
  -d '{
    "model_dir": "Trillim/BitNet-TRNQ",
    "lora_dir": "Trillim/BitNet-GenZ-LoRA-TRNQ"
  }'
```

Supported request fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `model_dir` | string | required store ID for the next base model |
| `num_threads` | int | optional runtime worker thread count |
| `lora_dir` | string | optional adapter store ID |
| `lora_quant` | string | optional LoRA runtime quantization |
| `unembed_quant` | string | optional unembedding quantization |
| `harness_name` | string | `default` or `search` |
| `search_provider` | string | `ddgs` or `brave` |
| `search_token_budget` | int | requested search-context budget |

Important behavior:

- Omitted init-time fields reset to Trillim defaults. They do not inherit the previous runtime's values.
- The effective search token budget is clamped to one quarter of the active model context window.
- Existing chat sessions become stale once swap handoff begins.
- `search_provider: "brave"` requires `SEARCH_API_KEY` in the server environment.

## Voice Routes

Voice routes exist only when the server includes `STT()` and `TTS()`. The CLI adds them with `--voice`.

Install the extra first:

```bash
uv add "trillim[voice]"
```

### `POST /v1/audio/transcriptions`

This is a raw-body route. Send audio bytes directly and set `content-type` to `audio/wav`, `audio/x-wav`, or `application/octet-stream`.

```bash
curl "http://127.0.0.1:8000/v1/audio/transcriptions?language=en" \
  -H "content-type: audio/wav" \
  --data-binary @recording.wav
```

Response:

```json
{"text": "transcribed text"}
```

Key facts:

- max upload size: `64 MiB`
- `language` is optional
- request bodies must contain raw `16-bit` little-endian mono `16 kHz` PCM or WAV that Trillim converts to that PCM format
- only one HTTP STT request is processed at a time; a concurrent request fails fast with `429`
- invalid `content-type`, invalid `content-length`, empty bodies, and mismatched body length are rejected before transcription starts

### `POST /v1/audio/speech`

This is also a raw-body route. Send UTF-8 text as the body.

```bash
curl -N http://127.0.0.1:8000/v1/audio/speech \
  -H "voice: alba" \
  -H "speed: 1.0" \
  --data-binary "Hello from Trillim."
```

Response format:

- `event: audio` with base64-encoded PCM chunks
- `event: done` when synthesis is complete
- `event: error` if the session fails after streaming has started

Important facts:

- the HTTP response is SSE, not WAV
- PCM is `24 kHz`, mono, `16-bit`
- max text body size: `6 MiB`
- only one live HTTP TTS request is allowed at a time across speech and voice-management routes

The Python SDK returns raw PCM through `TTSSession.collect(text)` or `TTSSession.synthesize(text)`.
If you need WAV, wrap the returned PCM in your application.

### `GET /v1/voices`

Lists built-in and custom voice names:

```bash
curl http://127.0.0.1:8000/v1/voices
```

### `POST /v1/voices`

Register a custom voice by sending raw audio bytes and a `name` header:

```bash
curl http://127.0.0.1:8000/v1/voices \
  -H "name: myvoice" \
  --data-binary @voice-sample.wav
```

Response:

```json
{"name": "myvoice", "status": "created"}
```

Important facts:

- max upload size: `10 MiB`
- max custom voices: `64`; built-in voices do not count against this quota
- max serialized custom voice state: `64 MiB`
- custom voice names and `voice` selectors accept only ASCII letters and digits
- custom voice storage lives under `~/.trillim/voices`
- new custom voices are persisted as Pocket TTS-native `.safetensors`
- legacy `.state` files and invalid safetensors files are skipped at startup with warnings and are not listed
- built-in voices cannot be shadowed; to replace a custom voice, delete it and then register the same name again
- while the server is running, the runtime voice cache is authoritative and disk storage is synchronized best-effort
- voice cloning support requires accepting the `kyutai/pocket-tts` terms and authenticating with Hugging Face
- if a reference sample exceeds the serialized voice-state cap, Trillim rejects it and you should retry with a shorter sample

One-time setup for custom voice cloning:

```bash
hf auth login
```

If `hf` is not installed globally, `uvx hf auth login` works as well.

### `DELETE /v1/voices/{voice_name}`

Delete a custom voice:

```bash
curl -X DELETE http://127.0.0.1:8000/v1/voices/myvoice
```

Deleting a built-in voice returns a validation error.
After deleting a custom voice, you may register the same name again with a new reference sample.

## Error Codes

The server maps the public failure modes consistently:

| Status | Meaning |
| --- | --- |
| `400` | invalid input, bad JSON, model mismatch, or validation failure |
| `409` | session conflict such as closed, stale, or exhausted chat state |
| `429` | component busy or not admitting more work |
| `504` | progress timeout |
| `503` | startup failure, worker failure, or other service-side error |

Use the SDK if you want the native Python exception types instead of HTTP status codes.
