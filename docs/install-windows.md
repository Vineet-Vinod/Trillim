# Install on Windows

## Prerequisites

- Python 3.12+
- x86_64 processor with AVX2

Check your Python version:

```powershell
python --version
```

If you don't have Python 3.12+, download it from [python.org](https://www.python.org/downloads/). During installation, make sure to check **"Add python.exe to PATH"**.

## Install with uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then create a project and install Trillim:

```powershell
uv init my-project
cd my-project
uv add trillim
```

With uv, all `trillim` commands must be prefixed with `uv run`:

```powershell
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

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install trillim
pip install trillim
```

Once activated, `trillim` is on your PATH:

```powershell
# List available models
trillim list

# Pull a pre-quantized model
trillim pull Trillim/BitNet-TRNQ

# Chat
trillim chat Trillim/BitNet-TRNQ

# Start the API server
trillim serve Trillim/BitNet-TRNQ
```

Remember to run `.venv\Scripts\activate` in each new terminal session.

### Global install

On Windows, pip installs globally by default (no externally-managed restriction):

```powershell
pip install trillim
```

`trillim` is then available in any terminal:

```powershell
trillim list
trillim pull Trillim/BitNet-TRNQ
trillim chat Trillim/BitNet-TRNQ
```
