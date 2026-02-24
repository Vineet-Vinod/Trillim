# Install on macOS

## Prerequisites

- Python 3.12+
- macOS with Apple Silicon (ARM64/NEON) or Intel (x86_64/AVX2)

Check your Python version:

```bash
python3 --version
```

If you don't have Python 3.12+, install it from [python.org](https://www.python.org/downloads/) or with Homebrew:

```bash
brew install python@3.12
```

## Install with uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create a project and install Trillim:

```bash
uv init my-project && cd my-project
uv add trillim
```

With uv, all `trillim` commands must be prefixed with `uv run`:

```bash
# List available models
uv run trillim list

# Pull a pre-quantized model
uv run trillim pull Trillim/BitNet-TRNQ

# Chat
uv run trillim chat Trillim/BitNet-TRNQ

# Start the API server
uv run trillim serve Trillim/BitNet-TRNQ
```

## Install with pip

### Virtual environment (recommended)

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install trillim
pip install trillim
```

Once activated, `trillim` is on your PATH:

```bash
# List available models
trillim list

# Pull a pre-quantized model
trillim pull Trillim/BitNet-TRNQ

# Chat
trillim chat Trillim/BitNet-TRNQ

# Start the API server
trillim serve Trillim/BitNet-TRNQ
```

Remember to run `source .venv/bin/activate` in each new terminal session.

### Global install

On macOS, Python 3.12+ marks the system Python as externally managed. To install globally you need either `--break-system-packages` or `pipx`:

```bash
# Option 1: override the restriction
pip install --break-system-packages trillim

# Option 2: use pipx (installs into its own isolated environment)
brew install pipx
pipx install trillim
```

With a global install (either method), `trillim` is available everywhere without activation:

```bash
trillim list
trillim pull Trillim/BitNet-TRNQ
trillim chat Trillim/BitNet-TRNQ
```
