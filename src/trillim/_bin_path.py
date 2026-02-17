# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Resolve paths to bundled C++ binaries."""

import os
import platform

_BIN_DIR = os.path.join(os.path.dirname(__file__), "_bin")


def _resolve(name: str) -> str:
    path = os.path.join(_BIN_DIR, name)
    if os.path.isfile(path):
        if not os.access(path, os.X_OK):
            os.chmod(path, os.stat(path).st_mode | 0o111)
        return path
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
