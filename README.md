# Trillim

Trillim is the platform for everything local AI. DarkNet is the CPU inference engine powering Trillim.

## Install

- Python 3.12+ required
- Linux also requires glibc 2.27+
- [uv](https://docs.astral.sh/uv/) is the recommended installer

Platform guides:

- [macOS](docs/install-mac.md)
- [Linux](docs/install-linux.md)
- [Windows](docs/install-windows.md)

If you installed with `uv`, prefix the CLI examples below with `uv run`.

## Common Workflows

### Pull a Model

```bash
trillim list
trillim pull Trillim/BitNet-TRNQ
```

### Chat in the Terminal

```bash
trillim chat Trillim/BitNet-TRNQ
```

`trillim chat` keeps multi-turn history, reuses the KV cache for follow-up turns, and supports `/new` to reset the conversation or `q` to quit.

### Search-Augmented Chat

Use the `search` harness with a search-tuned model:

```bash
trillim chat Trillim/BitNet-Search-TRNQ --harness search
```

DuckDuckGo (`ddgs`) is the default provider. To use Brave:

```bash
export SEARCH_API_KEY=<your_api_key>
trillim chat Trillim/BitNet-Search-TRNQ --harness search --search-provider brave
```

### Serve an OpenAI-Compatible API

Start the server:

```bash
trillim serve Trillim/BitNet-TRNQ
```

Main endpoints:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `POST /v1/models/load`

Example with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="BitNet-TRNQ",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

To switch a running server to the search harness, call `POST /v1/models/load` with `"harness": "search"` and optional `"search_provider": "ddgs" | "brave"`.

### Quantize a Model or Adapter

If you have a HuggingFace model with safetensors weights (currently only supports BitNet models):

```bash
# Quantize model weights -> qmodel.tensors + rope.cache
trillim quantize <path-to-model> --model

# Extract a PEFT LoRA adapter -> qmodel.lora
trillim quantize <path-to-model> --adapter <path-to-adapter>
```

### Use a LoRA Adapter

```bash
# Quantize a PEFT adapter into Trillim's format
trillim quantize <path-to-base-model> --adapter <path-to-adapter>

# Run the base model with the adapter
trillim chat Trillim/BitNet-TRNQ --lora <adapter-dir>

# Or pull a pre-quantized adapter and use it by ID
trillim pull Trillim/BitNet-GenZ-LoRA-TRNQ
trillim chat Trillim/BitNet-TRNQ --lora Trillim/BitNet-GenZ-LoRA-TRNQ
```

The same adapter settings can be changed at runtime through `POST /v1/models/load`.

### Runtime Quantization

Runtime quantization reduces memory use for selected layers during inference:

- `--lora-quant <type>` for LoRA layers: `none`, `bf16`, `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0`
- `--unembed-quant <type>` for the unembedding layer: `int8`, `q4_0`, `q5_0`, `q6_k`, `q8_0`

```bash
trillim chat Trillim/BitNet-TRNQ --lora <adapter-dir> --lora-quant int8
trillim chat Trillim/BitNet-TRNQ --unembed-quant q4_0
trillim serve Trillim/BitNet-TRNQ --lora-quant q8_0 --unembed-quant q4_0
```

### Voice Support

Install the optional `voice` extra before using speech endpoints:

```bash
uv add "trillim[voice]"
```

Or with `pip`:

```bash
pip install "trillim[voice]"
```

Then start the server with:

```bash
trillim serve Trillim/BitNet-TRNQ --voice
```

Voice endpoints:

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `GET /v1/voices`
- `POST /v1/voices`

Predefined voices are `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, and `azelma`.

For custom voice registration through `POST /v1/voices`, accept the terms for [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), create a HuggingFace token with `Read` access, and run:

```bash
hf auth login
```

Custom voice uploads through `POST /v1/voices` are limited to 8 MB per file.

That setup is only required once. Predefined voices work without it.

## Performance Highlights

Benchmark takeaways for DarkNet on consumer CPUs:

- Prefill throughput improvements are most visible when `num_threads >= 4`.
- Decode throughput is broadly comparable to bitnet.cpp on average, while DarkNet reaches higher peaks.
- Results are directional and depend on thermal behavior, boost policy, and memory bandwidth.

Prefill example:

![Prefill benchmark example](docs/imgs/Q4_0A.png)

Decode example:

![Decode benchmark example](docs/imgs/DecodeA.png)

## Supported Architectures

- `BitnetForCausalLM` for ternary BitNet models with ReLU² activation
- `LlamaForCausalLM` for Llama-style models with SiLU activation

## Platform Support

| Platform | Status |
|----------|--------|
| x86_64 (AVX2) | Supported |
| ARM64 (NEON) | Supported |

Thread count defaults to `num_cores - 2`. Override it with `--threads N`.

## Documentation

- [What Is Trillim?](docs/about-trillim.md)
- Install: [macOS](docs/install-mac.md), [Linux](docs/install-linux.md), [Windows](docs/install-windows.md)
- [CLI Reference](docs/cli.md)
- [Interactive Chat](docs/chat.md)
- [Python Components](docs/components.md)
- [API Server](docs/server.md)
- [Benchmarks](docs/benchmarks.md)

## License

The Trillim Python SDK source code is MIT-licensed. The C++ inference engine binaries (`inference`, `trillim-quantize`) bundled in the pip package are **proprietary**. You may use them as part of Trillim, but may not reverse-engineer or redistribute them separately. See [LICENSE](LICENSE) for the full terms.
