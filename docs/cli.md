# CLI Reference

Trillim provides a single `trillim` command with several subcommands for managing models, chatting, and running the API server.

## `trillim pull`

Download a pre-quantized model from HuggingFace.

```bash
trillim pull <model_id> [--revision <ref>] [--force]
```

| Flag | Description |
|---|---|
| `model_id` | HuggingFace model ID (e.g. `Trillim/BitNet-TRNQ`) |
| `--revision` | Branch, tag, or commit hash to download |
| `--force`, `-f` | Re-download even if the model already exists locally |

Models are stored in `~/.trillim/models/<org>/<model>/`.

### Example

```bash
trillim pull Trillim/BitNet-TRNQ
```

## `trillim chat`

Start an interactive chat session with a model. See [Chat](chat.md) for details on the chat interface.

```bash
trillim chat <model_dir> [options]
```

| Flag | Description |
|---|---|
| `model_dir` | Path to a local model directory, or a HuggingFace model ID (auto-resolved from `~/.trillim/models/`) |
| `--lora <dir>` | Path to a quantized LoRA adapter directory |
| `--threads <N>` | Number of inference threads. `0` (default) auto-detects as `num_cores - 2` |
| `--lora-quant <type>` | LoRA quantization level: `none`, `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0` |
| `--unembed-quant <type>` | Unembed layer quantization: `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0` |
| `--trust-remote-code` | Allow loading custom tokenizer code from the model directory |

### Examples

```bash
# Chat with a pulled model (resolved by HuggingFace ID)
trillim chat Trillim/BitNet-TRNQ

# Chat with a local model directory
trillim chat ./my-model-TRNQ

# Chat with a LoRA adapter
trillim chat Trillim/BitNet-TRNQ --lora Trillim/MyAdapter-TRNQ

# Use 4 threads
trillim chat Trillim/BitNet-TRNQ --threads 4
```

## `trillim serve`

Start an OpenAI-compatible API server. See [Server](server.md) for details on the API endpoints.

```bash
trillim serve <model_dir> [options]
```

| Flag | Description |
|---|---|
| `model_dir` | Path to a local model directory, or a HuggingFace model ID |
| `--host <addr>` | Bind address (default: `127.0.0.1`) |
| `--port <N>` | Bind port (default: `8000`) |
| `--voice` | Enable the voice pipeline (speech-to-text + text-to-speech) |
| `--whisper-model <size>` | Whisper model size (default: `base.en`) |
| `--voices-dir <dir>` | Directory for custom voice WAV files (default: `~/.trillim/voices`) |
| `--threads <N>` | Number of inference threads. `0` (default) auto-detects |
| `--lora-quant <type>` | LoRA quantization level |
| `--unembed-quant <type>` | Unembed layer quantization |
| `--trust-remote-code` | Allow loading custom tokenizer code |

### Examples

```bash
# Start the server
trillim serve Trillim/BitNet-TRNQ

# Start on a custom host and port
trillim serve Trillim/BitNet-TRNQ --host 0.0.0.0 --port 3000

# Start with the voice pipeline
trillim serve Trillim/BitNet-TRNQ --voice

# Start with a specific Whisper model
trillim serve Trillim/BitNet-TRNQ --voice --whisper-model medium.en
```

## `trillim quantize`

Quantize safetensors model weights and/or extract a LoRA adapter into Trillim's binary format.

```bash
trillim quantize <model_dir> [--model] [--adapter <dir>]
```

| Flag | Description |
|---|---|
| `model_dir` | Path to a HuggingFace model directory containing `config.json` and safetensors |
| `--model` | Quantize model weights. Outputs `<model_dir>-TRNQ/qmodel.tensors` and `rope.cache` |
| `--adapter <dir>` | Extract a PEFT LoRA adapter. Outputs `qmodel.lora` in `<adapter_dir>-TRNQ/` |

You can pass both `--model` and `--adapter` in the same command.

### Examples

```bash
# Quantize model weights
trillim quantize ./bitnet-2b --model

# Extract a LoRA adapter
trillim quantize ./bitnet-2b --adapter ./my-lora-checkpoint

# Both at once
trillim quantize ./bitnet-2b --model --adapter ./my-lora-checkpoint
```

## `trillim models`

List locally downloaded models and adapters.

```bash
trillim models [--json]
```

| Flag | Description |
|---|---|
| `--json` | Output as JSON instead of a formatted table |

### Example output

```
Models
MODEL ID              ARCH        SIZE  SOURCE
--------------------  ----------  ----  -----
Trillim/BitNet-TRNQ   BitNet      1.2G  microsoft/bitnet-b1.58-2B-4T-bf16

Adapters
ADAPTER ID                        SIZE  COMPATIBLE MODELS
------------------------------    ----  -----------------
Trillim/BitNet-GenZ-LoRA-TRNQ      24M  Trillim/BitNet-TRNQ
```

## `trillim list`

List models and adapters available on HuggingFace (from the Trillim organization).

```bash
trillim list [--json]
```

| Flag | Description |
|---|---|
| `--json` | Output as JSON instead of a formatted table |

Models that are already downloaded locally are marked with a `local` status.
