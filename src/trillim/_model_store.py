"""Shared model-store resolution policy for public Trillim entry points."""

from __future__ import annotations

import re
from pathlib import Path

MODELS_ROOT = Path.home() / ".trillim" / "models"
DOWNLOADED_ROOT = MODELS_ROOT / "Trillim"
LOCAL_ROOT = MODELS_ROOT / "Local"
STORE_ID_ERROR_MESSAGE = "Model IDs must use the form Trillim/<name> or Local/<name>"

_STORE_ID_RE = re.compile(r"^(Trillim|Local)/([A-Za-z0-9_.-]+)$")


def store_namespace_root(namespace: str) -> Path:
    """Return the namespace root for one supported store namespace."""
    if namespace == "Trillim":
        return DOWNLOADED_ROOT
    if namespace == "Local":
        return LOCAL_ROOT
    raise AssertionError(f"Unsupported namespace: {namespace}")


def parse_store_id(
    store_id: str | Path,
    *,
    error_type: type[Exception] = ValueError,
) -> tuple[str, str]:
    """Parse and validate one public Trillim store ID."""
    normalized = str(store_id).strip().replace("\\", "/")
    match = _STORE_ID_RE.fullmatch(normalized)
    if match is None:
        raise error_type(STORE_ID_ERROR_MESSAGE)
    namespace, name = match.groups()
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise error_type(STORE_ID_ERROR_MESSAGE)
    return namespace, name


def store_path_for_id(
    store_id: str | Path,
    *,
    error_type: type[Exception] = ValueError,
) -> Path:
    """Return the configured local path for one validated store ID."""
    namespace, name = parse_store_id(store_id, error_type=error_type)
    return store_namespace_root(namespace) / name


def resolve_existing_store_id(
    store_id: str | Path,
    *,
    error_type: type[Exception] = ValueError,
) -> Path:
    """Resolve one validated store ID to an existing local directory."""
    path = store_path_for_id(store_id, error_type=error_type)
    if not path.is_dir():
        namespace, _name = parse_store_id(store_id, error_type=error_type)
        raise error_type(
            f"Model '{str(store_id).strip()}' was not found locally in {store_namespace_root(namespace)}"
        )
    return path
