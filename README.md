# Trillim

Trillim is a local AI stack for CPUs. It gives you a CLI, a Python SDK, and a FastAPI server for running Trillim-formatted LLM bundles, plus optional speech-to-text and text-to-speech support.

Trillim supports both BitNet-style ternary bundles and PrismML's Bonsai (1-bit and ternary) bundles through the same managed model store and runtime surfaces.

DarkNet and the quantization tooling bundled with the package do the heavy inference work. The Python package is the orchestration layer around those binaries.

## Install

- Python 3.12 or newer is required.
- Linux wheels target `glibc >= 2.27`.
- `uv` is the recommended installer.
- Voice features require the optional `voice` extra.

Platform guides:

- [macOS](docs/install-mac.md)
- [Linux](docs/install-linux.md)
- [Windows](docs/install-windows.md)

If you install Trillim with `uv`, prefix CLI commands with `uv run`.

## Quick Start

Install the package:

```bash
uv add trillim
```

Pull a model and chat with it:

```bash
uv run trillim pull Trillim/BitNet-TRNQ
uv run trillim chat Trillim/BitNet-TRNQ
```

Start the local API server:

```bash
uv run trillim serve Trillim/BitNet-TRNQ
```

Use the Python SDK synchronously through `Runtime`:

```python
from trillim import LLM, Runtime

with Runtime(LLM("Trillim/BitNet-TRNQ")) as runtime:
    with runtime.llm.open_session() as session:
        reply = session.collect("Give me one sentence about local CPU inference.")
        print(reply)
```

## Common Workflows

### Pull and Inspect Bundles

`trillim models` lists bundles published by the `Trillim` Hugging Face organization. `trillim list` lists what you already have locally.

```bash
uv run trillim models
uv run trillim list
```

### Quantize a Local Model or Adapter

`trillim quantize` takes raw local filesystem paths and publishes the output under `~/.trillim/models/Local/`.

```bash
# Quantize a model bundle
uv run trillim quantize /path/to/model

# Quantize a LoRA adapter against its base model
uv run trillim quantize /path/to/base-model /path/to/adapter
```

Qwen3-based Bonsai checkpoints quantize into binary or grouped-ternary bundles, but Trillim still manages them under the same `Local/...-TRNQ` store naming and load flow.

### Use an Adapter

`chat` accepts an optional second positional argument for the adapter store ID:

```bash
uv run trillim chat Trillim/BitNet-TRNQ Trillim/BitNet-GenZ-LoRA-TRNQ
```

### Enable Voice Support

Install the extra first:

```bash
uv add "trillim[voice]"
```

Then start the voice-enabled server:

```bash
uv run trillim serve Trillim/BitNet-TRNQ --voice
```

## Documentation

### Learn

- [What Is Trillim?](docs/about-trillim.md)
- Install: [macOS](docs/install-mac.md), [Linux](docs/install-linux.md), [Windows](docs/install-windows.md)
- [CLI Reference](docs/cli.md)

### Extend and Serve

- [Python SDK](docs/components.md)
- [API Server](docs/server.md)

### Advanced

- [Advanced SDK and Server Notes](docs/advanced.md)
- [Benchmarks](docs/benchmarks.md)

## License

For the short license summary, see [What Is Trillim?](docs/about-trillim.md#license). Full terms are in [LICENSE](LICENSE).
