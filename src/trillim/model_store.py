# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Model store: pull pre-quantized models from HuggingFace and manage local copies."""

import json
import os
import platform
import re
from pathlib import Path

MODELS_DIR = Path.home() / ".trillim" / "models"
SUPPORTED_FORMAT_VERSION = 2

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
        raise RuntimeError(f"Model '{arg}' not found locally. Run: trillim pull {arg}")

    # Not a HF ID and not a directory — fall through to original path
    # (let downstream code raise its own error about the missing path)
    return arg


def validate_trillim_config(path: Path) -> dict | None:
    """Read trillim_config.json and warn about compatibility issues."""
    config_path = path / "trillim_config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, encoding="utf-8") as f:
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


class AdapterCompatError(Exception):
    """Raised when a LoRA adapter is incompatible with the base model."""


def validate_adapter_model_compat(adapter_dir: str, model_dir: str) -> None:
    """Check that a quantized LoRA adapter matches the base model.

    Raises ``AdapterCompatError`` if the adapter's ``trillim_config.json``
    is missing the v2 ``base_model_config_hash`` field (old format) or if
    the stored hash doesn't match the current base model.
    """
    from trillim.utils import compute_base_model_hash

    cfg_path = os.path.join(adapter_dir, "trillim_config.json")
    if not os.path.exists(cfg_path):
        return  # Absence is caught by the earlier file-existence check

    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return  # Unreadable config is warned about elsewhere

    fmt_ver = cfg.get("format_version", 1)
    stored_hash = cfg.get("base_model_config_hash")

    if fmt_ver < 2 or not stored_hash:
        raise AdapterCompatError(
            f"This adapter ({adapter_dir}) uses an older format (v{fmt_ver}) "
            "that is no longer supported.\n"
            "If you have the original LoRA weights, re-quantize with:\n"
            f"  trillim quantize {model_dir} --adapter <original_adapter_dir>\n"
            "Otherwise, download the latest version of this adapter."
        )

    current_hash = compute_base_model_hash(model_dir)
    if current_hash and current_hash != stored_hash:
        source = cfg.get("source_model", "unknown")
        raise AdapterCompatError(
            f"Adapter/model mismatch: this adapter was quantized for "
            f"'{source}' but the base model at '{model_dir}' has a "
            "different architecture config.\n"
            "Re-quantize with the correct base model:\n"
            f"  trillim quantize <correct_model_dir> --adapter <original_adapter_dir>"
        )


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
            raise RuntimeError(f"Repository '{model_id}' not found on HuggingFace") from e
        elif cls_name == "GatedRepoError":
            raise RuntimeError(
                f"'{model_id}' is a gated repository. Pass --token or run: hf auth login"
            ) from e
        else:
            raise

    print(f"Downloaded to {local_dir}")
    validate_trillim_config(local_dir)
    return local_dir


def _scan_models_dir() -> tuple[list[dict], list[dict]]:
    """Scan MODELS_DIR and return (models, adapters).

    A directory with ``config.json`` is a model.
    A directory *without* ``config.json`` but *with* ``trillim_config.json``
    is an adapter.
    """
    models: list[dict] = []
    adapters: list[dict] = []
    if not MODELS_DIR.is_dir():
        return models, adapters

    for org_dir in sorted(MODELS_DIR.iterdir()):
        if not org_dir.is_dir():
            continue
        for model_dir in sorted(org_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            has_config = (model_dir / "config.json").exists()
            tc_path = model_dir / "trillim_config.json"
            has_tc = tc_path.exists()

            if not has_config and not has_tc:
                continue

            entry_id = f"{org_dir.name}/{model_dir.name}"
            info: dict = {"model_id": entry_id, "path": str(model_dir)}

            # Read trillim_config.json for metadata
            if has_tc:
                try:
                    with open(tc_path, encoding="utf-8") as f:
                        tc = json.load(f)
                    info["architecture"] = tc.get("architecture", "")
                    info["source_model"] = tc.get("source_model", "")
                    info["quantization"] = tc.get("quantization", "")
                    info["base_model_config_hash"] = tc.get("base_model_config_hash", "")
                except (json.JSONDecodeError, OSError):
                    pass

            if has_config:
                # Compute config hash so adapters can be matched to this model
                from trillim.utils import compute_base_model_hash
                info["base_model_config_hash"] = compute_base_model_hash(str(model_dir))
                # Full model — report qmodel.tensors size
                tensors_path = model_dir / "qmodel.tensors"
                if tensors_path.exists():
                    size_bytes = tensors_path.stat().st_size
                    info["size_bytes"] = size_bytes
                    info["size_human"] = _human_size(size_bytes)
                else:
                    info["size_bytes"] = 0
                    info["size_human"] = "-"
                models.append(info)
            else:
                # Adapter only — report qmodel.lora size
                lora_path = model_dir / "qmodel.lora"
                if lora_path.exists():
                    size_bytes = lora_path.stat().st_size
                    info["size_bytes"] = size_bytes
                    info["size_human"] = _human_size(size_bytes)
                else:
                    info["size_bytes"] = 0
                    info["size_human"] = "-"
                adapters.append(info)

    return models, adapters


def list_models() -> list[dict]:
    """List all locally downloaded models."""
    models, _ = _scan_models_dir()
    return models


def list_adapters() -> list[dict]:
    """List all locally downloaded adapters (trillim_config.json but no config.json)."""
    _, adapters = _scan_models_dir()
    return adapters


def _human_size(n: int) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}".rstrip("0").rstrip(".")
        n /= 1024
    return f"{n:.1f} PB"
