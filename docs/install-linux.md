# Install on Linux

## Requirements

- x86_64 (AVX2) or ARM64 (NEON)
- Python 3.12 or newer
- glibc 2.27 or newer

Check glibc:

```bash
ldd --version | head -n 1
```

## 1. Check your Python version

Run:

```bash
python3 --version
```

If you see `Python 3.12.x` (or newer), continue to step 3.
If not, install/upgrade Python in step 2.

## 2. Install or upgrade to Python 3.12+

### Option A (recommended): install Python with uv

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, then install Python 3.12:

```bash
uv python install 3.12
uv run --python 3.12 python --version
```

### Option B: install Python from your distro packages

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip

# Fedora
sudo dnf install python3.12 python3.12-pip

# Arch Linux
sudo pacman -S python python-pip
```

Then verify:

```bash
python3 --version
```

## 3. Install Trillim with uv (recommended)

Create a project and add Trillim:

```bash
mkdir trillim-demo
cd trillim-demo
uv init .
uv python pin 3.12
uv add trillim
```

If you want voice features (`--voice`), install the voice extra:

```bash
uv add "trillim[voice]"
```

## 4. Verify the install

```bash
uv run python --version
uv run trillim --help
uv run trillim list
```

## 5. Run Trillim

With `uv`, prefix commands with `uv run`:

```bash
uv run trillim pull Trillim/BitNet-TRNQ
uv run trillim chat Trillim/BitNet-TRNQ
uv run trillim serve Trillim/BitNet-TRNQ
```

## 6. pip alternative (venv)

If you prefer `pip`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
pip install --upgrade pip
pip install trillim
```

Then run:

```bash
trillim --help
trillim list
```
