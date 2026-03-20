# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Resolve paths to bundled C++ binaries."""

import os
import platform
import sys
from pathlib import Path

_BIN_DIR = os.path.join(os.path.dirname(__file__), "_bin")
_EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
_SOURCE_BINARIES = {
    "inference": ("TRILLIM_INFERENCE_BIN", Path(__file__).resolve().parents[3] / "darknet" / "executables" / "trillim-inference"),
    "trillim-quantize": ("TRILLIM_QUANTIZE_BIN", Path(__file__).resolve().parents[3] / "darkquant" / "executables" / "trillim-quantize"),
}


def _ensure_executable(path: str) -> str:
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | 0o111)
    return path


def _resolve(name: str) -> str:
    env_var, source_path = _SOURCE_BINARIES.get(name, (None, None))
    if env_var:
        override = os.environ.get(env_var)
        if override:
            if not os.path.isfile(override):
                raise RuntimeError(f"{env_var} is set, but no binary exists at {override}")
            return _ensure_executable(override)

    path = os.path.join(_BIN_DIR, name + _EXE_SUFFIX)
    if os.path.isfile(path):
        return _ensure_executable(path)
    # Fall back to the name without suffix (e.g. WSL or Cygwin)
    alt_path = os.path.join(_BIN_DIR, name)
    if alt_path != path and os.path.isfile(alt_path):
        return _ensure_executable(alt_path)
    if source_path and source_path.is_file():
        return _ensure_executable(str(source_path))
    if not os.listdir(_BIN_DIR) or os.listdir(_BIN_DIR) == [".gitkeep"]:
        raise RuntimeError(
            "No packaged binaries found. Install from PyPI, set the TRILLIM_*_BIN "
            "environment variables, or build the sibling darkquant/darknet "
            "executables in this source tree."
        )
    raise RuntimeError(
        f"Binary '{name}' not found at {path}. "
        f"Platform ({platform.machine()}) may not be supported."
    )


def inference_bin() -> str:
    return _resolve("inference")


def quantize_bin() -> str:
    return _resolve("trillim-quantize")
