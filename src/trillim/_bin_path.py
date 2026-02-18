# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Resolve paths to bundled C++ binaries."""

import os
import platform
import sys

_BIN_DIR = os.path.join(os.path.dirname(__file__), "_bin")
_EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""


def _resolve(name: str) -> str:
    path = os.path.join(_BIN_DIR, name + _EXE_SUFFIX)
    if os.path.isfile(path):
        if not os.access(path, os.X_OK):
            os.chmod(path, os.stat(path).st_mode | 0o111)
        return path
    # Fall back to the name without suffix (e.g. WSL or Cygwin)
    alt_path = os.path.join(_BIN_DIR, name)
    if alt_path != path and os.path.isfile(alt_path):
        if not os.access(alt_path, os.X_OK):
            os.chmod(alt_path, os.stat(alt_path).st_mode | 0o111)
        return alt_path
    if not os.listdir(_BIN_DIR) or os.listdir(_BIN_DIR) == [".gitkeep"]:
        raise RuntimeError(
            "No binaries found. It looks like trillim was built from source, "
            "which is not supported — the C++ inference engine is distributed "
            "as pre-compiled binaries via PyPI. Install with: pip install trillim"
        )
    raise RuntimeError(
        f"Binary '{name}' not found at {path}. "
        f"Platform ({platform.machine()}) may not be supported."
    )


def inference_bin() -> str:
    return _resolve("inference")


def quantize_bin() -> str:
    return _resolve("trillim-quantize")
