# Trillim

High-performance CPU inference engine for BitNet models. Runs ternary-quantized models ({-1, 0, 1} weights) using platform-specific SIMD optimizations (AVX2 on x86, NEON on ARM).

## Quick Start

### Prerequisites

- Python 3.12+ with [`uv`](https://github.com/astral-sh/uv) - can use pip or any package manager

### Install and run

```bash
# Install trillim
uv add trillim

# Pull a pre-quantized model
uv run trillim pull Trillim/BitNet-TRNQ

# Chat
uv run trillim chat Trillim/BitNet-TRNQ
```

### Quantize your own model

If you have a HuggingFace BitNet model with safetensors weights:

```bash
# Quantize model weights → qmodel.tensors + rope.cache
uv run trillim quantize <path-to-model> --model

# Optionally extract a PEFT LoRA adapter → qmodel.lora
uv run trillim quantize <path-to-model> --adapter <path-to-adapter>
```

## API Server

Trillim includes an OpenAI-compatible API server:

```bash
# Start the server
uv run trillim serve <model-dir>

# With voice pipeline (speech-to-text + text-to-speech)
uv run trillim serve <model-dir> --voice
```

Endpoints:
- `POST /v1/chat/completions` — chat completions (streaming supported)
- `POST /v1/completions` — text completions
- `GET /v1/models` — list loaded models
- `POST /v1/models/load` — hot-swap models and LoRA adapters at runtime
- `POST /v1/audio/transcriptions` — speech-to-text (with `--voice`)
- `POST /v1/audio/speech` — text-to-speech (with `--voice`)
- `GET /v1/voices` — list available TTS voices
- `POST /v1/voices` — register a custom voice from audio (need to accept pocket-tts' terms on huggingface)

## Python SDK

The server is built on a composable SDK. Each capability (LLM, Whisper, TTS) is a standalone component:

```python
from trillim import Server, LLM, TTS, Whisper

# Inference only
Server(LLM("models/BitNet")).run()

# Inference + voice
Server(LLM("models/BitNet"), Whisper(), TTS()).run()

# TTS only
Server(TTS()).run()
```

## LoRA Adapters

Trillim supports PEFT LoRA adapters as bf16 corrections on top of the ternary base model:

```bash
# Ensure qmodel.lora is in the directory 
# (uv run trillim quantize ... will do this)
uv run trillim chat Trillim/BitNet-TRNQ --lora
```

## Supported Architectures

- `BitnetForCausalLM` — BitNet with ternary weights and ReLU² activation
- `LlamaForCausalLM` — Llama-style with SiLU activation

## Platform Support

| Platform | Status |
|----------|--------|
| x86_64 (AVX2) | Supported |
| ARM64 (NEON) | Supported |

Thread count is auto-detected as `num_cores - 2`. Override by passing a `--threads N` CLI argument.

## License

The Trillim Python SDK source code is MIT-licensed. The C++ inference engine binaries (`inference`, `trillim-quantize`) bundled in the pip package are **proprietary** — you may use them as part of Trillim but may not reverse-engineer or redistribute them separately. See [LICENSE](LICENSE) for full terms.
