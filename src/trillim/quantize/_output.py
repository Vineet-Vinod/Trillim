"""Managed Local bundle output, recovery, and copy policy."""

from __future__ import annotations

import ast
import json
import os
import shutil
from collections import deque
from importlib import metadata as importlib_metadata
from pathlib import Path
import tomllib

from trillim import _model_store
from trillim._bundle_metadata import (
    CURRENT_FORMAT_VERSION,
    canonicalize_model_config,
    compute_base_model_config_hash,
)

from ._config import ModelQuantizeConfig

_MODEL_ALLOWLIST = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
)
_ADAPTER_EXCLUDED_NAMES = {
    ".quantize_manifest.bin",
    "qmodel.lora",
    "qmodel.tensors",
    "rope.cache",
    "trillim_config.json",
}
_PUBLISH_COMPLETE_MARKER = ".trillim-quantize-complete"
_SUPPORTED_PLATFORMS = ["x86_64", "aarch64"]
_MAX_REMOTE_CODE_DEPTH = 16
_MAX_REMOTE_CODE_FILES = 64
_MAX_REMOTE_CODE_BYTES = 4 * 1024 * 1024


def prepare_output_target(source_dir: Path) -> Path:
    _model_store.LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    preferred = _model_store.LOCAL_ROOT / f"{source_dir.name}-TRNQ"
    recover_publish_state(preferred)
    if not preferred.exists():
        return preferred
    if _should_prompt_for_overwrite() and _confirm_overwrite(preferred):
        return preferred
    return _allocate_dedup_target(preferred)


def build_staging_dir(target: Path) -> Path:
    staging = target.parent / f"{target.name}-new"
    if staging.exists():
        raise RuntimeError(f"Staging directory is still present after recovery: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def mark_staging_complete(staging_dir: Path) -> None:
    (staging_dir / _PUBLISH_COMPLETE_MARKER).write_text("ready\n", encoding="utf-8")


def publish_staging_dir(target: Path) -> None:
    staging = target.parent / f"{target.name}-new"
    backup = target.parent / f"{target.name}-old"
    if not staging.is_dir():
        raise RuntimeError(f"Staging directory not found: {staging}")
    if not _is_complete_staging_dir(staging):
        raise RuntimeError(f"Staging directory is incomplete: {staging}")
    moved_old = False
    if target.exists():
        os.replace(target, backup)
        moved_old = True
    try:
        os.replace(staging, target)
    except Exception:
        if moved_old and not target.exists() and backup.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def recover_publish_state(target: Path) -> None:
    staging = target.parent / f"{target.name}-new"
    backup = target.parent / f"{target.name}-old"
    _require_directory_or_missing(staging)
    _require_directory_or_missing(backup)
    _require_directory_or_missing(target)
    if target.exists():
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists():
            shutil.rmtree(backup)
        return
    if backup.exists():
        if staging.exists() and _is_complete_staging_dir(staging):
            os.replace(staging, target)
            shutil.rmtree(backup)
            return
        if staging.exists():
            shutil.rmtree(staging)
        os.replace(backup, target)
        return
    if staging.exists():
        if _is_complete_staging_dir(staging):
            os.replace(staging, target)
        else:
            shutil.rmtree(staging)


def copy_model_support_files(model_dir: Path, output_dir: Path) -> None:
    for filename in _MODEL_ALLOWLIST:
        source_path = model_dir / filename
        if source_path.is_file():
            _copy_file(source_path, output_dir / filename)
    for relative_path in _collect_remote_code_files(model_dir):
        _copy_file(model_dir / relative_path, output_dir / relative_path)


def copy_adapter_support_files(adapter_dir: Path, output_dir: Path) -> None:
    for source_path in sorted(adapter_dir.rglob("*")):
        if source_path.is_dir():
            continue
        relative_path = source_path.relative_to(adapter_dir)
        if _should_skip_adapter_path(relative_path):
            continue
        _copy_file(source_path, output_dir / relative_path)


def write_model_metadata(
    output_dir: Path,
    *,
    config: ModelQuantizeConfig,
    model_dir: Path,
) -> None:
    payload = {
        "trillim_version": _project_version(),
        "format_version": CURRENT_FORMAT_VERSION,
        "type": "model",
        "quantization": "ternary",
        "source_model": config.source_model,
        "architecture": config.arch_name,
        "platforms": list(_SUPPORTED_PLATFORMS),
        "base_model_config_hash": compute_base_model_config_hash(model_dir),
        "remote_code": _has_remote_code_references(model_dir),
    }
    _write_json(output_dir / "trillim_config.json", payload)


def write_adapter_metadata(
    output_dir: Path,
    *,
    config: ModelQuantizeConfig,
    adapter_dir: Path,
    model_dir: Path,
) -> None:
    adapter_config_path = adapter_dir / "adapter_config.json"
    source_model = ""
    if adapter_config_path.is_file():
        payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        source_model = str(payload.get("base_model_name_or_path", ""))
    payload = {
        "trillim_version": _project_version(),
        "format_version": CURRENT_FORMAT_VERSION,
        "type": "lora_adapter",
        "quantization": "ternary",
        "source_model": source_model,
        "architecture": config.arch_name,
        "platforms": list(_SUPPORTED_PLATFORMS),
        "base_model_config_hash": compute_base_model_config_hash(model_dir),
        "remote_code": _has_remote_code_references(adapter_dir),
    }
    _write_json(output_dir / "trillim_config.json", payload)


def _allocate_dedup_target(preferred: Path) -> Path:
    suffix = 2
    while True:
        candidate = preferred.parent / f"{preferred.name}-{suffix}"
        recover_publish_state(candidate)
        if not candidate.exists():
            return candidate
        suffix += 1


def _should_prompt_for_overwrite() -> bool:
    return sys_stdin_isatty() and sys_stdout_isatty()


def _confirm_overwrite(target: Path) -> bool:
    while True:
        response = input(f"Overwrite existing bundle {target.name}? [y/n] ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False


def sys_stdin_isatty() -> bool:
    return bool(getattr(__import__("sys").stdin, "isatty", lambda: False)())


def sys_stdout_isatty() -> bool:
    return bool(getattr(__import__("sys").stdout, "isatty", lambda: False)())


def _copy_file(source_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)


def _should_skip_adapter_path(relative_path: Path) -> bool:
    if "__pycache__" in relative_path.parts:
        return True
    name = relative_path.name
    if name in _ADAPTER_EXCLUDED_NAMES:
        return True
    if name.endswith(".pyc") or name.endswith(".tmp"):
        return True
    if name == "adapter_model.safetensors":
        return True
    if name.startswith("adapter_model.safetensors."):
        return True
    if name == "adapter_model.bin":
        return True
    return False


def _require_directory_or_missing(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise RuntimeError(f"Managed quantize state is not a directory: {path}")


def _is_complete_staging_dir(path: Path) -> bool:
    return (path / _PUBLISH_COMPLETE_MARKER).is_file()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _project_version() -> str:
    try:
        return importlib_metadata.version("trillim")
    except importlib_metadata.PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        return str(payload["project"]["version"])


def _collect_remote_code_files(model_dir: Path) -> list[Path]:
    metadata = _load_remote_code_metadata(model_dir)
    queue = deque((_parse_remote_code_module_path(ref), 0) for ref in _collect_remote_code_class_refs(metadata))
    collected: list[Path] = []
    seen: set[Path] = set()
    total_bytes = 0
    while queue:
        relative_path, depth = queue.popleft()
        if relative_path in seen:
            continue
        if depth > _MAX_REMOTE_CODE_DEPTH:
            raise ValueError("Remote-code import graph exceeds the supported depth")
        source_path = model_dir / relative_path
        if not source_path.is_file():
            raise ValueError(f"Remote-code module not found: {relative_path}")
        if len(seen) >= _MAX_REMOTE_CODE_FILES:
            raise ValueError("Remote-code import graph exceeds the supported file budget")
        total_bytes += source_path.stat().st_size
        if total_bytes > _MAX_REMOTE_CODE_BYTES:
            raise ValueError("Remote-code import graph exceeds the supported byte budget")
        seen.add(relative_path)
        collected.append(relative_path)
        for module_name in _relative_import_module_names(source_path):
            queue.append(
                (
                    _resolve_relative_import_module_path(
                        source_relative_path=relative_path,
                        module_name=module_name,
                        model_dir=model_dir,
                    ),
                    depth + 1,
                )
            )
    return collected


def _has_remote_code_references(model_dir: Path) -> bool:
    return bool(_collect_remote_code_class_refs(_load_remote_code_metadata(model_dir)))


def _load_remote_code_metadata(model_dir: Path) -> dict[str, dict | None]:
    config_payload = _load_optional_json(model_dir / "config.json")
    if config_payload is not None:
        config_payload = canonicalize_model_config(config_payload)
    return {
        "config": config_payload,
        "tokenizer_config": _load_optional_json(model_dir / "tokenizer_config.json"),
    }


def _load_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _collect_remote_code_class_refs(metadata: dict[str, dict | None]) -> list[str]:
    refs: list[str] = []
    for payload, key in (
        (metadata["tokenizer_config"], "AutoTokenizer"),
        (metadata["config"], "AutoTokenizer"),
        (metadata["config"], "AutoConfig"),
    ):
        refs.extend(_extract_auto_map_refs(payload, key=key))
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        deduped.append(ref)
    return deduped


def _extract_auto_map_refs(payload: dict | None, *, key: str) -> list[str]:
    if not isinstance(payload, dict):
        return []
    auto_map = payload.get("auto_map")
    if key == "AutoTokenizer" and isinstance(auto_map, (list, tuple)):
        values = auto_map
    elif isinstance(auto_map, dict):
        value = auto_map.get(key)
        values = value if isinstance(value, (list, tuple)) else [value]
    else:
        return []
    return [value for value in values if isinstance(value, str) and value]


def _parse_remote_code_module_path(class_ref: str) -> Path:
    if "--" in class_ref:
        raise ValueError(
            "External remote-code repositories are currently unsupported for local model bundles"
        )
    module_name, _, class_name = class_ref.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Remote-code class reference is currently unsupported: {class_ref}")
    if "." in module_name:
        raise ValueError(f"Package-scoped remote-code entry points are currently unsupported: {class_ref}")
    return Path(f"{module_name}.py")


def _relative_import_module_names(module_path: Path) -> list[str]:
    content = module_path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(module_path))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level > 1:
            raise ValueError(
                f"Parent relative imports in remote-code modules are currently unsupported: {module_path}"
            )
        if node.level != 1:
            continue
        if node.module and "." in node.module:
            raise ValueError(
                f"Package-scoped relative imports in remote-code modules are currently unsupported: {module_path}"
            )
        if node.module:
            names.append(node.module)
            continue
        for alias in node.names:
            if alias.name != "*":
                names.append(alias.name)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _resolve_relative_import_module_path(
    *,
    source_relative_path: Path,
    module_name: str,
    model_dir: Path,
) -> Path:
    relative_module_path = source_relative_path.parent / Path(*[part for part in module_name.split(".") if part]).with_suffix(".py")
    package_init_path = source_relative_path.parent / module_name / "__init__.py"
    if (model_dir / package_init_path).is_file():
        raise ValueError(
            "Package-scoped relative imports in remote-code modules are currently unsupported: "
            f"{source_relative_path}"
        )
    return relative_module_path
