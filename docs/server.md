# API Server

Use `trillim serve` when you want local OpenAI-compatible HTTP endpoints.

If you installed with `uv`, prefix each command on this page with `uv run`.

## Before You Start

- Pull a model first with `trillim pull Trillim/BitNet-TRNQ`
- If you want voice endpoints, install `uv add "trillim[voice]"` or `pip install "trillim[voice]"`
- If you want Brave search, set `SEARCH_API_KEY`
- If you want custom voice registration through `POST /v1/voices`, accept the terms for [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), create a HuggingFace token with `Read` access, and run `hf auth login` once

## Start the Server

Start the default server:

```bash
trillim serve Trillim/BitNet-TRNQ
```

By default the server binds to `127.0.0.1:8000`. Override host and port with:

```bash
trillim serve Trillim/BitNet-TRNQ --host 0.0.0.0 --port 3000
```

Enable speech-to-text and text-to-speech with:

```bash
trillim serve Trillim/BitNet-TRNQ --voice
```

## Change the Loaded Model or Harness

`trillim serve` starts with the default harness. Use `POST /v1/models/load` to swap models, load a LoRA adapter, or switch to the search harness without restarting the server.

Switch models:

```bash
curl http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "Trillim/BitNet-TRNQ"
  }'
```

Enable the search harness on a running server:

```bash
curl http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "Trillim/BitNet-Search-TRNQ",
    "harness": "search",
    "search_provider": "ddgs"
  }'
```

If you use `"search_provider": "brave"`, set:

```bash
export SEARCH_API_KEY=<your_api_key>
```

## Endpoint Summary

| Endpoint | Purpose | Notes |
|---|---|---|
| `POST /v1/chat/completions` | Chat completions | Streaming supported |
| `POST /v1/completions` | Raw text completions | Streaming supported |
| `GET /v1/models` | Show the loaded model | Always available |
| `POST /v1/models/load` | Swap model, adapter, or harness | Always available |
| `POST /v1/audio/transcriptions` | Speech-to-text | Requires `--voice` |
| `POST /v1/audio/speech` | Text-to-speech | Requires `--voice` |
| `GET /v1/voices` | List voices | Requires `--voice` |
| `POST /v1/voices` | Register a custom voice | Requires `--voice` |
| `DELETE /v1/voices/{voice_id}` | Delete a custom voice | Requires `--voice` |

## `POST /v1/chat/completions`

Send a conversation and get a model response. This endpoint is compatible with the OpenAI chat completions API.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Request body:

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | array | required | List of `{"role": "...", "content": "..."}` objects |
| `model` | string | `""` | Model identifier for client compatibility |
| `temperature` | float | model default | Sampling temperature, `>= 0` |
| `top_k` | int | model default | Top-K sampling, `>= 1` |
| `top_p` | float | model default | Nucleus sampling threshold, `(0, 1]` |
| `max_tokens` | int | null | Maximum tokens to generate |
| `repetition_penalty` | float | model default | Repetition penalty, `>= 0` |
| `stream` | bool | `false` | Enable server-sent event streaming |

When the active harness is `search`, this endpoint can run multi-step search-augmented generation for search-tuned models.

If the rendered prompt reaches the model context limit, the server returns `400` with a detail message describing the token count and context window.

Response:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "BitNet-TRNQ",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "The capital of France is Paris."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20,
    "cached_tokens": 0
  }
}
```

### Streaming

Set `"stream": true` to receive server-sent events:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"BitNet-TRNQ","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"BitNet-TRNQ","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"BitNet-TRNQ","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Even with the `search` harness enabled, streamed deltas contain assistant text only. Internal search progress is not emitted on this OpenAI-compatible endpoint.

## `POST /v1/completions`

Send a raw prompt without a chat template.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown fox"
  }'
```

This endpoint uses the same sampling fields as `/v1/chat/completions`, but replaces `messages` with `prompt` and returns `text` instead of `message`.

Streaming is supported with `"stream": true`.

`/v1/completions` does not use chat harness orchestration.

Requests that exceed the active model context window return `400`.

## `GET /v1/models`

Return the currently loaded model:

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "BitNet-TRNQ", "object": "model", "created": 0, "owned_by": "local"}
  ]
}
```

## `POST /v1/models/load`

Swap to a different model, LoRA adapter, or harness configuration at runtime. Models and adapters must already exist under `~/.trillim/models/`.

```bash
curl http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "Trillim/BitNet-TRNQ"
  }'
```

Request body:

| Field | Type | Default | Description |
|---|---|---|---|
| `model_dir` | string | required | HuggingFace model ID or path under `~/.trillim/models/` |
| `adapter_dir` | string | null | LoRA adapter ID or path under `~/.trillim/models/` |
| `harness` | string | null | `default` or `search` |
| `search_provider` | string | null | `ddgs` or `brave` when `harness` is `search` |
| `threads` | int | null | `null` keeps the current setting, `0` auto-detects |
| `lora_quant` | string | null | LoRA quantization level |
| `unembed_quant` | string | null | Unembed quantization level |

Response:

```json
{
  "status": "success",
  "model": "BitNet-TRNQ",
  "recompiled": false,
  "message": ""
}
```

## Voice Endpoints

These routes are only available when the server starts with `--voice`.

### `POST /v1/audio/transcriptions`

Speech-to-text using Whisper. Upload an audio file as multipart form data. The maximum file size is 8 MB.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-1
```

Form fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file to transcribe |
| `model` | string | `"whisper-1"` | Model identifier for client compatibility |
| `language` | string | null | Optional language hint such as `"en"` |
| `response_format` | string | `"json"` | `"json"` or `"text"` |

Response:

```json
{
  "text": "Hello, how are you?"
}
```

### `POST /v1/audio/speech`

Convert text to speech. The response is a WAV or PCM audio stream. Speech starts streaming progressively even when `speed` is set, with bounded lookahead rather than full-utterance buffering.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "alba", "speed": 1.5}' \
  --output speech.wav
```

Request body:

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Text to synthesize |
| `voice` | string | `"alba"` | Voice ID |
| `speed` | number | `1.0` | Pitch-preserving playback speed, from `0.25` to `4.0` |
| `response_format` | string | `"wav"` | `"wav"` or `"pcm"` |

`speed` is fixed for the lifetime of a single `/v1/audio/speech` request. Dynamic mid-stream speed changes are only available through the direct Python `TTS` component session API.

### `GET /v1/voices`

List all available voices, including the built-in set and any saved custom voices.

```bash
curl http://localhost:8000/v1/voices
```

Built-in voices: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

### `POST /v1/voices`

Upload an audio sample to register a custom voice. The sample is saved to the configured voices directory and persists across server restarts.

```bash
curl http://localhost:8000/v1/voices \
  -F voice_id=my-voice \
  -F file=@sample.wav
```

Form fields:

| Field | Type | Description |
|---|---|---|
| `voice_id` | string | Identifier for the new voice |
| `file` | file | Audio sample, max 8 MB |

If custom voice registration fails, verify that you completed the HuggingFace setup in the prerequisites at the top of this page.

### `DELETE /v1/voices/{voice_id}`

Delete a previously registered custom voice. Built-in voices cannot be deleted.

```bash
curl -X DELETE http://localhost:8000/v1/voices/my-voice
```

## OpenAI Python Client

You can use the official OpenAI client library against the local server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="BitNet-TRNQ",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Streaming works too:

```python
stream = client.chat.completions.create(
    model="BitNet-TRNQ",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```
