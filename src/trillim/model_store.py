# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Model store: pull pre-quantized models from HuggingFace and manage local copies."""

import json
import os
import platform
import re
import sys
from pathlib import Path

MODELS_DIR = Path.home() / ".trillim" / "models"
SUPPORTED_FORMAT_VERSION = 1

_HF_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _looks_like_hf_id(arg: str) -> bool:
    """Return True if arg looks like a HuggingFace model ID (org/name)."""
    if arg.startswith(("/", ".", "~")):
        return False
    return bool(_HF_ID_RE.match(arg))


def resolve_model_dir(arg: str) -> str:
    """Resolve a model argument to a local directory path.

    If *arg* is an existing directory, return it unchanged.  If it looks like
    a HuggingFace model ID, look it up in ``~/.trillim/models/``.
    """
    if os.path.isdir(arg):
        return arg

    # Expand ~ just in case
    expanded = os.path.expanduser(arg)
    if os.path.isdir(expanded):
        return expanded

    if _looks_like_hf_id(arg):
        local = MODELS_DIR / arg
        if local.is_dir():
            return str(local)
        print(f"Error: Model '{arg}' not found locally.\nRun: trillim pull {arg}")
        sys.exit(1)

    # Not a HF ID and not a directory — fall through to original path
    # (let downstream code raise its own error about the missing path)
    return arg


def validate_trillim_config(path: Path) -> dict | None:
    """Read trillim_config.json and warn about compatibility issues."""
    config_path = path / "trillim_config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not read trillim_config.json: {e}")
        return None

    fmt_ver = config.get("format_version")
    if fmt_ver is not None and fmt_ver > SUPPORTED_FORMAT_VERSION:
        print(
            f"Warning: Model format version {fmt_ver} is newer than supported "
            f"version {SUPPORTED_FORMAT_VERSION}. Consider upgrading trillim."
        )

    platforms = config.get("platforms")
    if platforms and platform.machine() not in platforms:
        print(
            f"Warning: This model lists platforms {platforms} but your system "
            f"is {platform.machine()}. It may not work correctly."
        )

    return config


def pull_model(model_id: str, revision: str | None = None, token: str | None = None, force: bool = False) -> Path:
    """Download a pre-quantized model from HuggingFace."""
    local_dir = MODELS_DIR / model_id

    if local_dir.is_dir() and not force:
        print(f"Model '{model_id}' already exists at {local_dir}")
        print("Use --force to re-download.")
        return local_dir

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pulling {model_id} ...")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            model_id,
            local_dir=str(local_dir),
            revision=revision,
            token=token,
        )
    except Exception as e:
        cls_name = type(e).__name__
        if cls_name == "RepositoryNotFoundError":
            print(f"Error: Repository '{model_id}' not found on HuggingFace.")
            sys.exit(1)
        elif cls_name == "GatedRepoError":
            print(
                f"Error: '{model_id}' is a gated repository. "
                "Pass --token or run: huggingface-cli login"
            )
            sys.exit(1)
        else:
            raise

    print(f"Downloaded to {local_dir}")
    validate_trillim_config(local_dir)
    return local_dir


def list_models() -> list[dict]:
    """List all locally downloaded models."""
    models = []
    if not MODELS_DIR.is_dir():
        return models

    for org_dir in sorted(MODELS_DIR.iterdir()):
        if not org_dir.is_dir():
            continue
        for model_dir in sorted(org_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            # Must have config.json to be a valid model
            if not (model_dir / "config.json").exists():
                continue

            model_id = f"{org_dir.name}/{model_dir.name}"
            info: dict = {"model_id": model_id, "path": str(model_dir)}

            # Read trillim_config.json for metadata
            tc_path = model_dir / "trillim_config.json"
            if tc_path.exists():
                try:
                    with open(tc_path) as f:
                        tc = json.load(f)
                    info["architecture"] = tc.get("architecture", "")
                    info["source_model"] = tc.get("source_model", "")
                    info["quantization"] = tc.get("quantization", "")
                except (json.JSONDecodeError, OSError):
                    pass

            # Report qmodel.tensors size
            tensors_path = model_dir / "qmodel.tensors"
            if tensors_path.exists():
                size_bytes = tensors_path.stat().st_size
                info["size_bytes"] = size_bytes
                info["size_human"] = _human_size(size_bytes)
            else:
                info["size_bytes"] = 0
                info["size_human"] = "-"

            models.append(info)

    return models


def _human_size(n: int) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}".rstrip("0").rstrip(".")
        n /= 1024
    return f"{n:.1f} PB"
