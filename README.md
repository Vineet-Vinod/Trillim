# Trillim

[What is Trillim?](docs/about-trillim.md)

## Quick Start

### Installation

- Python 3.12+ required
- Install with [uv](https://docs.astral.sh/uv/) (recommended) or pip

Pick your platform for full instructions:

- [macOS](docs/install-mac.md)
- [Linux](docs/install-linux.md)
- [Windows](docs/install-windows.md)

> **Note:** The rest of this README shows bare `trillim` commands. If you're using uv, prefix each command with `uv run` (e.g. `uv run trillim chat ...`).

### Quantize your own model

If you have a HuggingFace BitNet model with safetensors weights:

```bash
# Quantize model weights → qmodel.tensors + rope.cache
trillim quantize <path-to-model> --model

# Optionally extract a PEFT LoRA adapter → qmodel.lora
trillim quantize <path-to-model> --adapter <path-to-adapter>
```

## Chat

Start an interactive conversation in your terminal:

```bash
trillim chat Trillim/BitNet-TRNQ
```

Multi-turn conversations are supported with automatic prompt caching for fast follow-ups. Use `/new` to start a fresh conversation, or `q` to quit.

See the [Chat guide](docs/chat.md) for details on LoRA adapters, sampling parameters, and performance tips.

## API Server

Trillim includes an OpenAI-compatible API server:

```bash
# Start the server
trillim serve Trillim/BitNet-TRNQ

# With voice pipeline (speech-to-text + text-to-speech)
trillim serve Trillim/BitNet-TRNQ --voice
```

Endpoints:
- `POST /v1/chat/completions` — chat completions (streaming supported)
- `POST /v1/completions` — text completions
- `GET /v1/models` — list loaded models
- `POST /v1/models/load` — hot-swap models and LoRA adapters at runtime
- `POST /v1/audio/transcriptions` — speech-to-text (with `--voice`)
- `POST /v1/audio/speech` — text-to-speech (with `--voice`)
- `GET /v1/voices` — list available TTS voices
- `POST /v1/voices` — register a custom voice from audio (see [Voice Cloning Setup](#voice-cloning-setup))

Works with the OpenAI Python client out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="BitNet-TRNQ",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

See the [Server guide](docs/server.md) for full endpoint documentation, request/response schemas, the Python SDK, and voice pipeline usage.

## LoRA Adapters

Trillim supports PEFT LoRA adapters as bf16 corrections on top of the ternary base model. The adapter lives in its own directory (separate from the base model) and must be quantized first:

```bash
# Quantize a PEFT adapter into Trillim's format
trillim quantize <path-to-base-model> --adapter <path-to-adapter>

# Chat with the base model + adapter
trillim chat Trillim/BitNet-TRNQ --lora <adapter-dir>

# Or pull a pre-quantized adapter and use it by ID
trillim pull Trillim/BitNet-GenZ-LoRA-TRNQ
trillim chat Trillim/BitNet-TRNQ --lora Trillim/BitNet-GenZ-LoRA-TRNQ
```

Adapters can also be hot-swapped at runtime via the API server's `POST /v1/models/load` endpoint. See the [Server guide](docs/server.md) for details.

## Runtime Quantization

Separately from the offline `trillim quantize` step (which converts model weights to ternary), Trillim can quantize specific layers at inference time to reduce memory usage. This is controlled with two flags available on both `chat` and `serve`:

- **`--lora-quant <type>`** — quantize LoRA adapter layers. Options: `none`, `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0`. Only applies when using `--lora`.
- **`--unembed-quant <type>`** — quantize the unembedding (output projection) layer. Options: `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0`.

```bash
# Quantize LoRA layers to int8 for lower memory
trillim chat Trillim/BitNet-TRNQ --lora <adapter-dir> --lora-quant int8

# Quantize the unembed layer to q4_0
trillim chat Trillim/BitNet-TRNQ --unembed-quant q4_0

# Both at once
trillim serve Trillim/BitNet-TRNQ --lora-quant q8_0 --unembed-quant q4_0
```

Lower quantization levels (e.g. `q4_0`) use less memory at a small quality cost. These options can also be set per-request when hot-swapping models via `POST /v1/models/load`. See the [CLI reference](docs/cli.md) for the full flag list.

## Voice Cloning Setup

The voice pipeline (`--voice`) includes 8 predefined voices that work out of the box: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`.

To register **custom voices** (voice cloning via `POST /v1/voices`), you need to accept the PocketTTS model terms and authenticate with HuggingFace:

1. Go to [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) on HuggingFace and accept the model's terms.
2. Create a token on HuggingFace (under Access Tokens) with `Read` permissions.
2. Log in locally so the token is available to download the voice cloning weights:

```bash
hf auth login
```

This only needs to be done once. After that, custom voice registration works automatically. If you skip this step, you'll get an error when trying to register a custom voice — predefined voices will still work fine.

## Supported Architectures

- `BitnetForCausalLM` — BitNet with ternary weights and ReLU² activation
- `LlamaForCausalLM` — Llama-style with SiLU activation

## Platform Support

| Platform | Status |
|----------|--------|
| x86_64 (AVX2) | Supported |
| ARM64 (NEON) | Supported |

Thread count is auto-detected as `num_cores - 2`. Override by passing a `--threads N` CLI argument.

## Documentation

- [What is Trillim?](docs/about-trillim.md) — overview, motivation, and who it's for
- Install — [macOS](docs/install-mac.md) | [Linux](docs/install-linux.md) | [Windows](docs/install-windows.md)
- [CLI Reference](docs/cli.md) — all commands and flags
- [Chat](docs/chat.md) — interactive chat interface
- [Server](docs/server.md) — API endpoints, Python SDK, and OpenAI client usage

## License

The Trillim Python SDK source code is MIT-licensed. The C++ inference engine binaries (`inference`, `trillim-quantize`) bundled in the pip package are **proprietary** — you may use them as part of Trillim but may not reverse-engineer or redistribute them separately. See [LICENSE](LICENSE) for full terms.
