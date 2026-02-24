# API Server

Trillim includes an OpenAI-compatible API server built on FastAPI. You can use it with any OpenAI client library or make direct HTTP requests.

## Starting the Server

```bash
trillim serve <model_dir>
```

By default the server binds to `127.0.0.1:8000`. Override with `--host` and `--port`:

```bash
trillim serve Trillim/BitNet-TRNQ --host 0.0.0.0 --port 3000
```

To enable the voice pipeline (speech-to-text and text-to-speech):

```bash
trillim serve Trillim/BitNet-TRNQ --voice
```

## Endpoints

### Chat Completions

**`POST /v1/chat/completions`**

Send a conversation and get a model response. Compatible with the OpenAI chat completions API.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | array | required | List of `{"role": "...", "content": "..."}` objects |
| `model` | string | `""` | Model identifier (informational) |
| `temperature` | float | model default | Sampling temperature (>= 0) |
| `top_k` | int | model default | Top-K sampling (>= 1) |
| `top_p` | float | model default | Nucleus sampling threshold (0, 1] |
| `max_tokens` | int | null | Maximum tokens to generate |
| `repetition_penalty` | float | model default | Repetition penalty (>= 0) |
| `stream` | bool | `false` | Enable server-sent events streaming |

**Response:**

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

#### Streaming

Set `"stream": true` to receive server-sent events. Each event contains a delta chunk:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"BitNet-TRNQ","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"BitNet-TRNQ","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"BitNet-TRNQ","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Text Completions

**`POST /v1/completions`**

Raw text completion without a chat template.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown fox"
  }'
```

The request and response fields match the chat completions endpoint, except `messages` is replaced by a `prompt` string and the response uses `text` instead of `message`.

Streaming is supported with `"stream": true`.

### List Models

**`GET /v1/models`**

Returns the currently loaded model.

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

### Hot-Swap Models

**`POST /v1/models/load`**

Switch to a different model or LoRA adapter at runtime without restarting the server. Only models stored in `~/.trillim/models/` are allowed (use `trillim pull` first).

```bash
curl http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "Trillim/BitNet-TRNQ"
  }'
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `model_dir` | string | required | HuggingFace model ID or path under `~/.trillim/models/` |
| `adapter_dir` | string | null | LoRA adapter directory to load |
| `threads` | int | null | Thread count override. `null` keeps the current setting, `0` auto-detects |
| `lora_quant` | string | null | LoRA quantization level |
| `unembed_quant` | string | null | Unembed quantization level |

**Response:**

```json
{
  "status": "success",
  "model": "BitNet-TRNQ",
  "recompiled": false,
  "message": ""
}
```

## Voice Endpoints

These endpoints are only available when the server is started with `--voice`.

### Transcribe Audio

**`POST /v1/audio/transcriptions`**

Speech-to-text using Whisper. Upload an audio file (max 8 MB) as multipart form data.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-1
```

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file to transcribe |
| `model` | string | `"whisper-1"` | Model identifier (informational) |
| `language` | string | null | Language hint (e.g. `"en"`) |
| `response_format` | string | `"json"` | `"json"` or `"text"` |

**Response:**

```json
{
  "text": "Hello, how are you?"
}
```

### Text-to-Speech

**`POST /v1/audio/speech`**

Convert text to audio. Returns a WAV audio stream.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "alba"}' \
  --output speech.wav
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Text to synthesize |
| `voice` | string | `"alba"` | Voice to use |
| `response_format` | string | `"wav"` | `"wav"` or `"pcm"` |

### List Voices

**`GET /v1/voices`**

List all available voices (predefined and custom).

```bash
curl http://localhost:8000/v1/voices
```

**Predefined voices:** `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

### Register a Custom Voice

**`POST /v1/voices`**

Upload an audio sample to create a custom voice. The audio is saved to the voices directory and persists across server restarts.

```bash
curl http://localhost:8000/v1/voices \
  -F voice_id=my-voice \
  -F file=@sample.wav
```

**Form fields:**

| Field | Type | Description |
|---|---|---|
| `voice_id` | string | Identifier for the new voice |
| `file` | file | Audio sample (max 8 MB) |

### Delete a Custom Voice

**`DELETE /v1/voices/{voice_id}`**

Remove a previously registered custom voice. Predefined voices cannot be deleted.

```bash
curl -X DELETE http://localhost:8000/v1/voices/my-voice
```

## Python SDK

The server is built on a composable SDK. Each capability is a standalone component that you can use programmatically:

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

### LLM Component

```python
from trillim import LLM

LLM(
    model_dir="~/.trillim/models/Trillim/BitNet-TRNQ",
    adapter_dir=None,          # optional LoRA adapter path
    num_threads=0,             # 0 = auto-detect
    trust_remote_code=False,
    lora_quant=None,           # "none", "int8", "q4_0", etc.
    unembed_quant=None,        # "int8", "q4_0", etc.
)
```

### Whisper (Speech to Text) Component

```python
from trillim import Whisper

Whisper(
    model_size="base.en",   # Whisper model size
    compute_type="int8",
    cpu_threads=2,
)
```

### TTS (Text to Speech) Component

```python
from trillim import TTS

TTS(
    voices_dir="~/.trillim/voices",  # where custom voices are stored
)
```

### Custom Routes

Access the underlying FastAPI app to add custom routes:

```python
from trillim import Server, LLM

server = Server(LLM("~/.trillim/models/Trillim/BitNet-TRNQ"))
app = server.app

@app.get("/health")
async def health():
    return {"status": "ok"}

server.run(host="0.0.0.0", port=8000)
```

## Using with the OpenAI Python Client

Since the API is OpenAI-compatible, you can use the official OpenAI client library:

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
