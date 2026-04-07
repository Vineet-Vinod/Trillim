"""Model directory validation and metadata extraction for LLMs."""

from __future__ import annotations

import ast
import errno
import json
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from trillim._bundle_metadata import (
    CURRENT_FORMAT_VERSION,
    canonicalize_model_config as _canonicalize_model_config,
    compute_base_model_config_hash as _compute_base_model_config_hash,
)
from trillim.components.llm._config import (
    ActivationType,
    ArchitectureType,
    InitConfig,
    ModelRuntimeConfig,
)
from trillim.errors import ModelValidationError

_STOP_TOKEN_NAMES = (
    "<|eot_id|>",
    "<|im_end|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
)
_DEFAULT_EOS_TOKENS = {
    ArchitectureType.BITNET: 128009,
    ArchitectureType.LLAMA: 128009,
    ArchitectureType.QWEN35: 248044,
}
_MODEL_RUNTIME_ARTIFACTS = ("qmodel.tensors", "rope.cache")
_LORA_RUNTIME_ARTIFACTS = ("qmodel.lora",)
_TOKENIZER_FALLBACK_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
)
_MAX_REMOTE_CODE_DEPTH = 16
_MAX_REMOTE_CODE_FILES = 64
_MAX_REMOTE_CODE_BYTES = 4 * 1024 * 1024
_SUPPORTED_ADAPTER_FORMAT_VERSION = CURRENT_FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class _ArchitectureInfo:
    arch_type: ArchitectureType
    activation: ActivationType
    has_attn_sub_norm: bool
    has_ffn_sub_norm: bool
    has_qkv_bias: bool = False


_ARCH_REGISTRY: dict[str, _ArchitectureInfo] = {
    "bitnetforcausallm": _ArchitectureInfo(
        arch_type=ArchitectureType.BITNET,
        activation=ActivationType.RELU_SQR,
        has_attn_sub_norm=True,
        has_ffn_sub_norm=True,
    ),
    "llamaforcausallm": _ArchitectureInfo(
        arch_type=ArchitectureType.LLAMA,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    ),
    "qwen3_5forconditionalgeneration": _ArchitectureInfo(
        arch_type=ArchitectureType.QWEN35,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    ),
    "bitnetbpeforcausallm": _ArchitectureInfo(
        arch_type=ArchitectureType.BITNET,
        activation=ActivationType.RELU_SQR,
        has_attn_sub_norm=True,
        has_ffn_sub_norm=True,
    ),
    "qwen3forcausallm": _ArchitectureInfo(
        arch_type=ArchitectureType.BONSAI,
        activation=ActivationType.SILU,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    ),
}
_ACTIVATION_MAP = {
    "relu_squared": ActivationType.RELU_SQR,
    "relu2": ActivationType.RELU_SQR,
    "relu_sqr": ActivationType.RELU_SQR,
    "silu": ActivationType.SILU,
    "swish": ActivationType.SILU,
}


@dataclass(slots=True)
class RuntimeFiles:
    """Validated runtime paths and optional overlay state."""

    model_dir: Path
    metadata_dir: Path
    adapter_dir: Path | None = None
    _temp_dir: TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        """Release any temporary overlay directory."""
        if self._temp_dir is None:
            return
        self._temp_dir.cleanup()
        self._temp_dir = None


@dataclass(frozen=True, slots=True)
class _OverlayMetadata:
    config: dict
    added_tokens: dict | list | None
    generation_config: dict | None
    special_tokens_map: dict | None
    tokenizer_config: dict | None


def validate_model_dir(
    model_dir: str | Path,
    *,
    metadata_dir: str | Path | None = None,
) -> ModelRuntimeConfig:
    """Validate a model directory and extract runtime metadata."""
    path = _resolve_directory(
        model_dir,
        label="Model directory",
        symlink_message="Model directory must not use symlinks",
    )
    metadata_path = (
        path
        if metadata_dir is None
        else _resolve_directory(
            metadata_dir,
            label="Metadata directory",
            symlink_message="Metadata directory must not use symlinks",
        )
    )
    _validate_model_bundle_metadata(path)
    config_path = metadata_path / "config.json"
    _raise_if_symlink(config_path, "Model bundle must not use symlinks")
    if not config_path.is_file():
        raise ModelValidationError(f"config.json not found in {metadata_path}")
    _require_runtime_artifacts(path)
    config_payload = _load_json(config_path)
    if not isinstance(config_payload, dict):
        raise ModelValidationError(
            f"config.json must be a JSON object in {metadata_path}"
        )
    config = _canonicalize_model_config(config_payload)
    arch_info = _resolve_arch_info(config)
    dimensions = _extract_dimensions(config)
    eos_tokens = _collect_eos_tokens(config, arch_info.arch_type, metadata_path)
    return ModelRuntimeConfig(
        name=path.name,
        path=path,
        arch_type=arch_info.arch_type,
        activation=_resolve_activation(config, arch_info),
        hidden_dim=dimensions["hidden_dim"],
        intermediate_dim=dimensions["intermediate_dim"],
        num_layers=dimensions["num_layers"],
        num_heads=dimensions["num_heads"],
        num_kv_heads=dimensions["num_kv_heads"],
        vocab_size=dimensions["vocab_size"],
        head_dim=dimensions["head_dim"],
        max_position_embeddings=dimensions["max_position_embeddings"],
        norm_eps=float(config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-6))),
        rope_theta=_resolve_rope_theta(config),
        eos_tokens=tuple(eos_tokens),
        has_qkv_bias=_resolve_qkv_bias(config, arch_info),
        tie_word_embeddings=_resolve_tied_embeddings(config),
        has_attn_sub_norm=arch_info.has_attn_sub_norm,
        has_ffn_sub_norm=arch_info.has_ffn_sub_norm,
    )


def validate_lora_dir(
    lora_dir: str | Path,
    *,
    model_dir: str | Path | None = None,
) -> Path:
    """Validate a LoRA directory path."""
    path = _resolve_directory(
        lora_dir,
        label="LoRA directory",
        symlink_message="LoRA directory must not use symlinks",
    )
    _require_adapter_artifacts(path)
    adapter_config = _load_adapter_config(path)
    _validate_adapter_metadata(path, adapter_config=adapter_config)
    if model_dir is not None:
        _validate_adapter_compatibility(
            path,
            adapter_config=adapter_config,
            model_dir=Path(model_dir),
        )
    return path


def prepare_runtime_files(
    init_config: InitConfig,
    *,
    trust_remote_code: bool,
) -> RuntimeFiles:
    """Prepare validated runtime paths and an optional lora-first overlay."""
    model_dir = _resolve_directory(
        init_config.model_dir,
        label="Model directory",
        symlink_message="Model directory must not use symlinks",
    )
    _require_runtime_artifacts(model_dir)
    if init_config.lora_dir is None:
        return RuntimeFiles(model_dir=model_dir, metadata_dir=model_dir)
    adapter_dir = validate_lora_dir(init_config.lora_dir, model_dir=model_dir)
    _ensure_overlay_filesystem_supported(model_dir, adapter_dir)
    temp_dir = TemporaryDirectory(prefix="trillim-llm-")
    overlay_dir = Path(temp_dir.name)
    try:
        _build_overlay_dir(
            overlay_dir,
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        temp_dir.cleanup()
        raise
    return RuntimeFiles(
        model_dir=model_dir,
        metadata_dir=overlay_dir,
        adapter_dir=adapter_dir,
        _temp_dir=temp_dir,
    )


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"Could not read JSON from {path}") from exc


def _validate_model_bundle_metadata(model_dir: Path) -> None:
    config_path = model_dir / "trillim_config.json"
    _raise_if_symlink(config_path, "Model bundle must not use symlinks")
    if not config_path.is_file():
        raise ModelValidationError(
            f"Model bundle metadata is missing or unsupported in {model_dir}"
        )
    payload = _load_json(config_path)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != CURRENT_FORMAT_VERSION
    ):
        raise ModelValidationError(
            f"Model bundle metadata is missing or unsupported in {model_dir}"
        )


def _resolve_arch_info(config: dict) -> _ArchitectureInfo:
    architectures = config.get("architectures", [])
    arch_name = architectures[0] if architectures else "unknown"
    try:
        return _ARCH_REGISTRY[arch_name.lower()]
    except KeyError as exc:
        raise ModelValidationError(
            f"Unsupported model architecture: {arch_name}"
        ) from exc


def _require_runtime_artifacts(model_dir: Path) -> None:
    for filename in _MODEL_RUNTIME_ARTIFACTS:
        artifact_path = model_dir / filename
        _raise_if_symlink(artifact_path, "Model bundle must not use symlinks")
        if not artifact_path.is_file():
            raise ModelValidationError(f"{filename} not found in {model_dir}")


def _require_adapter_artifacts(adapter_dir: Path) -> None:
    for filename in _LORA_RUNTIME_ARTIFACTS:
        artifact_path = adapter_dir / filename
        _raise_if_symlink(artifact_path, "LoRA directory must not use symlinks")
        if not artifact_path.is_file():
            raise ModelValidationError(f"{filename} not found in {adapter_dir}")


def _load_adapter_config(adapter_dir: Path) -> dict:
    config_path = adapter_dir / "trillim_config.json"
    _raise_if_symlink(config_path, "LoRA directory must not use symlinks")
    if not config_path.is_file():
        raise ModelValidationError(f"trillim_config.json not found in {adapter_dir}")
    payload = _load_json(config_path)
    if not isinstance(payload, dict):
        raise ModelValidationError(
            f"Adapter metadata must be a JSON object in {config_path}"
        )
    return payload


def _validate_adapter_metadata(
    adapter_dir: Path,
    *,
    adapter_config: dict,
) -> None:
    format_version = adapter_config.get("format_version", 1)
    stored_hash = adapter_config.get("base_model_config_hash")
    if (
        format_version != _SUPPORTED_ADAPTER_FORMAT_VERSION
        or not isinstance(stored_hash, str)
        or not stored_hash
    ):
        raise ModelValidationError(
            f"Adapter compatibility metadata is missing or unsupported in {adapter_dir}"
        )


def _validate_adapter_compatibility(
    adapter_dir: Path,
    *,
    adapter_config: dict,
    model_dir: Path,
) -> None:
    stored_hash = adapter_config["base_model_config_hash"]
    try:
        current_hash = _compute_base_model_config_hash(model_dir)
    except ValueError as exc:
        raise ModelValidationError(
            f"Could not validate adapter compatibility against {model_dir}"
        ) from exc
    if current_hash != stored_hash:
        source_model = adapter_config.get("source_model")
        detail = (
            "this adapter was quantized for a different base model"
            if not isinstance(source_model, str) or not source_model
            else f"this adapter was quantized for {source_model!r}"
        )
        raise ModelValidationError(
            f"Adapter/model mismatch: {detail}"
        )


def _extract_dimensions(config: dict) -> dict[str, int]:
    hidden_dim = _require_positive_int(config.get("hidden_size"), "hidden_size")
    intermediate_dim = _require_positive_int(
        config.get("intermediate_size"),
        "intermediate_size",
    )
    num_heads = _require_positive_int(
        config.get("num_attention_heads"),
        "num_attention_heads",
    )
    num_kv_heads = _require_positive_int(
        config.get("num_key_value_heads", num_heads),
        "num_key_value_heads",
    )
    max_position_embeddings = _require_positive_int(
        config.get("max_position_embeddings", 4096),
        "max_position_embeddings",
    )
    vocab_size = _require_positive_int(config.get("vocab_size"), "vocab_size")
    head_dim = int(config.get("head_dim", hidden_dim // num_heads))
    if head_dim <= 0:
        raise ModelValidationError("head_dim must be a positive integer")
    return {
        "hidden_dim": _align_to_128(hidden_dim),
        "intermediate_dim": _align_to_128(intermediate_dim),
        "num_layers": _require_positive_int(config.get("num_hidden_layers"), "num_hidden_layers"),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
    }


def _align_to_128(value: int) -> int:
    return ((value + 127) // 128) * 128


def _require_positive_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ModelValidationError(f"{field_name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ModelValidationError(f"{field_name} must be a positive integer")
    return number


def _resolve_activation(config: dict, arch_info: _ArchitectureInfo) -> ActivationType:
    hidden_act = config.get("hidden_act")
    if hidden_act is None:
        return arch_info.activation
    try:
        return _ACTIVATION_MAP[str(hidden_act).lower()]
    except KeyError as exc:
        raise ModelValidationError(
            f"Unsupported activation function: {hidden_act}"
        ) from exc


def _resolve_rope_theta(config: dict) -> float:
    rope_theta = config.get("rope_theta")
    if rope_theta is None:
        rope_parameters = config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", 10000.0)
        else:
            rope_theta = 10000.0
    try:
        return float(rope_theta)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError("rope_theta must be numeric") from exc


def _resolve_qkv_bias(
    config: dict,
    arch_info: _ArchitectureInfo,
) -> bool:
    return bool(config.get("attention_bias", arch_info.has_qkv_bias))


def _resolve_tied_embeddings(config: dict) -> bool:
    return bool(config.get("tie_word_embeddings", False))


def _collect_eos_tokens(
    config: dict,
    arch_type: ArchitectureType,
    metadata_dir: Path,
) -> list[int]:
    eos_raw = config.get("eos_token_id", _DEFAULT_EOS_TOKENS.get(arch_type, 2))
    if isinstance(eos_raw, list):
        eos_tokens = [int(token_id) for token_id in eos_raw]
    else:
        eos_tokens = [int(eos_raw)]
    tokenizer_payload = _load_optional_json(metadata_dir / "tokenizer.json")
    added_tokens_payload = _load_optional_json(metadata_dir / "added_tokens.json")
    eos_tokens.extend(_collect_added_tokens(tokenizer_payload))
    eos_tokens.extend(_collect_added_tokens(added_tokens_payload))
    deduped: list[int] = []
    seen: set[int] = set()
    for token_id in eos_tokens:
        if token_id in seen:
            continue
        seen.add(token_id)
        deduped.append(token_id)
    if not deduped:
        raise ModelValidationError("No EOS tokens could be determined for the model")
    return deduped


def _load_optional_json(path: Path) -> dict | None:
    _raise_if_symlink(path, "Model bundle must not use symlinks")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_directory(
    directory: str | Path,
    *,
    label: str,
    symlink_message: str,
) -> Path:
    original = str(directory)
    path = Path(os.path.abspath(Path(directory).expanduser()))
    _raise_if_symlink(path, symlink_message)
    if not path.exists():
        raise ModelValidationError(f"{label} does not exist: {original}")
    if not path.is_dir():
        raise ModelValidationError(f"{label} is not a directory: {path}")
    return path


def _raise_if_symlink(path: Path, message: str) -> None:
    if path.is_symlink():
        raise ModelValidationError(f"{message}: {path}")


def _build_overlay_dir(
    overlay_dir: Path,
    *,
    model_dir: Path,
    adapter_dir: Path,
    trust_remote_code: bool,
) -> None:
    metadata = _build_overlay_metadata(model_dir, adapter_dir)
    for filename in _MODEL_RUNTIME_ARTIFACTS:
        _materialize_required_file(
            overlay_dir,
            source_dir=model_dir,
            relative_path=Path(filename),
            symlink_message="Model bundle must not use symlinks",
            mode="hardlink",
        )
    for filename in _LORA_RUNTIME_ARTIFACTS:
        _materialize_required_file(
            overlay_dir,
            source_dir=adapter_dir,
            relative_path=Path(filename),
            symlink_message="LoRA directory must not use symlinks",
            mode="hardlink",
        )
    _write_json_file(overlay_dir / "config.json", metadata.config)
    _write_optional_json_file(overlay_dir / "added_tokens.json", metadata.added_tokens)
    _write_optional_json_file(overlay_dir / "generation_config.json", metadata.generation_config)
    _write_optional_json_file(overlay_dir / "special_tokens_map.json", metadata.special_tokens_map)
    _write_optional_json_file(overlay_dir / "tokenizer_config.json", metadata.tokenizer_config)
    for filename in _TOKENIZER_FALLBACK_FILES:
        _materialize_fallback_file(
            overlay_dir,
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            relative_path=Path(filename),
            mode="copy",
        )
    if trust_remote_code:
        for relative_path in _collect_remote_code_files(model_dir, adapter_dir, metadata):
            _materialize_fallback_file(
                overlay_dir,
                model_dir=model_dir,
                adapter_dir=adapter_dir,
                relative_path=relative_path,
                mode="copy",
            )


def _build_overlay_metadata(model_dir: Path, adapter_dir: Path) -> _OverlayMetadata:
    base_config = _load_required_json_strict(
        model_dir / "config.json",
        symlink_message="Model bundle must not use symlinks",
        transform=_canonicalize_model_config,
    )
    adapter_config = _load_optional_json_strict(
        adapter_dir / "config.json",
        symlink_message="LoRA directory must not use symlinks",
        transform=_canonicalize_model_config,
    )
    base_tokenizer_config = _load_optional_json_strict(
        model_dir / "tokenizer_config.json",
        symlink_message="Model bundle must not use symlinks",
    )
    adapter_tokenizer_config = _load_optional_json_strict(
        adapter_dir / "tokenizer_config.json",
        symlink_message="LoRA directory must not use symlinks",
    )
    adapter_has_explicit_auto_tokenizer = _adapter_declares_auto_tokenizer(
        adapter_config,
        adapter_tokenizer_config,
    )
    return _OverlayMetadata(
        config=_merge_tokenizer_loader_payloads(
            base_config,
            adapter_config,
            adapter_has_explicit_auto_tokenizer=adapter_has_explicit_auto_tokenizer,
        )
        or {},
        added_tokens=_merge_json_payloads(
            _load_optional_json_strict(
                model_dir / "added_tokens.json",
                symlink_message="Model bundle must not use symlinks",
            ),
            _load_optional_json_strict(
                adapter_dir / "added_tokens.json",
                symlink_message="LoRA directory must not use symlinks",
            ),
        ),
        generation_config=_merge_json_payloads(
            _load_optional_json_with_message(
                model_dir / "generation_config.json",
                symlink_message="Model bundle must not use symlinks",
            ),
            _load_optional_json_with_message(
                adapter_dir / "generation_config.json",
                symlink_message="LoRA directory must not use symlinks",
            ),
        ),
        special_tokens_map=_merge_json_payloads(
            _load_optional_json_strict(
                model_dir / "special_tokens_map.json",
                symlink_message="Model bundle must not use symlinks",
            ),
            _load_optional_json_strict(
                adapter_dir / "special_tokens_map.json",
                symlink_message="LoRA directory must not use symlinks",
            ),
        ),
        tokenizer_config=_merge_tokenizer_loader_payloads(
            base_tokenizer_config,
            adapter_tokenizer_config,
            adapter_has_explicit_auto_tokenizer=adapter_has_explicit_auto_tokenizer,
        ),
    )


def _materialize_required_file(
    overlay_dir: Path,
    *,
    source_dir: Path,
    relative_path: Path,
    symlink_message: str,
    mode: str,
) -> None:
    source_path = _resolve_used_path(source_dir, relative_path, symlink_message)
    if not source_path.is_file():
        raise ModelValidationError(f"{relative_path.name} not found in {source_dir}")
    _materialize_file(overlay_dir / relative_path, source_path, mode=mode)


def _materialize_fallback_file(
    overlay_dir: Path,
    *,
    model_dir: Path,
    adapter_dir: Path,
    relative_path: Path,
    mode: str,
) -> None:
    source_path = _locate_overlay_source(model_dir, adapter_dir, relative_path)
    if source_path is None:
        return
    _materialize_file(overlay_dir / relative_path, source_path, mode=mode)


def _locate_overlay_source(
    model_dir: Path,
    adapter_dir: Path,
    relative_path: Path,
) -> Path | None:
    for source_dir, symlink_message in (
        (adapter_dir, "LoRA directory must not use symlinks"),
        (model_dir, "Model bundle must not use symlinks"),
    ):
        source_path = _resolve_used_path(source_dir, relative_path, symlink_message)
        if source_path.is_file():
            return source_path
    return None


def _resolve_used_path(source_dir: Path, relative_path: Path, symlink_message: str) -> Path:
    current = source_dir
    for part in relative_path.parts:
        current = current / part
        _raise_if_symlink(current, symlink_message)
    return current


def _materialize_file(destination: Path, source_path: Path, *, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source_path, destination)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                raise ModelValidationError(
                    f"Could not hardlink runtime artifact across filesystems: {source_path}"
                ) from exc
            raise ModelValidationError(f"Could not hardlink runtime artifact: {source_path}") from exc
        return
    shutil.copy2(source_path, destination)


def _ensure_overlay_filesystem_supported(model_dir: Path, adapter_dir: Path) -> None:
    temp_root = Path(tempfile.gettempdir())
    devices = {
        _filesystem_device(model_dir),
        _filesystem_device(adapter_dir),
        _filesystem_device(temp_root),
    }
    if len(devices) != 1:
        raise ModelValidationError(
            "LoRA overlays do not support multi-filesystem deployments; "
            "model_dir, lora_dir, and the process temp directory must share one filesystem"
        )


def _filesystem_device(path: Path) -> int:
    try:
        return path.stat().st_dev
    except OSError as exc:
        raise ModelValidationError(f"Could not inspect filesystem for {path}") from exc


def _write_json_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_optional_json_file(path: Path, payload: dict | list | None) -> None:
    if payload is None:
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _merge_json_payloads(base: dict | list | None, override: dict | list | None) -> dict | list | None:
    if base is None:
        if override is None:
            return None
        return dict(override) if isinstance(override, dict) else list(override) if isinstance(override, list) else override
    if override is None:
        return dict(base) if isinstance(base, dict) else list(base) if isinstance(base, list) else base
    if not isinstance(base, dict) or not isinstance(override, dict):
        return dict(override) if isinstance(override, dict) else list(override) if isinstance(override, list) else override
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_json_payloads(merged[key], value)
        else:
            merged[key] = value
    return merged


def _adapter_declares_auto_tokenizer(*payloads: dict | None) -> bool:
    return any(_extract_auto_map_refs(payload, key="AutoTokenizer") for payload in payloads)


def _merge_tokenizer_loader_payloads(
    base: dict | None,
    override: dict | None,
    *,
    adapter_has_explicit_auto_tokenizer: bool,
) -> dict | None:
    merged = _merge_json_payloads(base, override)
    if not isinstance(merged, dict) or adapter_has_explicit_auto_tokenizer:
        return merged
    _restore_base_tokenizer_loader_fields(merged, base)
    return merged


def _restore_base_tokenizer_loader_fields(merged: dict, base: dict | None) -> None:
    tokenizer_class = None
    if isinstance(base, dict):
        candidate = base.get("tokenizer_class")
        if isinstance(candidate, str) and candidate:
            tokenizer_class = candidate
    if tokenizer_class is None:
        merged.pop("tokenizer_class", None)
    else:
        merged["tokenizer_class"] = tokenizer_class
    _restore_base_auto_map_entry(merged, base, key="AutoTokenizer")


def _restore_base_auto_map_entry(merged: dict, base: dict | None, *, key: str) -> None:
    base_auto_map = base.get("auto_map") if isinstance(base, dict) else None
    merged_auto_map = merged.get("auto_map")
    base_value, base_has_value, base_uses_list = _extract_auto_map_value(base, key=key)
    if base_uses_list:
        merged["auto_map"] = base_value
        return
    if isinstance(base_auto_map, dict):
        updated_auto_map = (
            dict(merged_auto_map) if isinstance(merged_auto_map, dict) else dict(base_auto_map)
        )
        if base_has_value:
            updated_auto_map[key] = base_value
        else:
            updated_auto_map.pop(key, None)
        if updated_auto_map:
            merged["auto_map"] = updated_auto_map
        else:
            merged.pop("auto_map", None)
        return
    if isinstance(merged_auto_map, dict):
        if key not in merged_auto_map:
            return
        updated_auto_map = dict(merged_auto_map)
        updated_auto_map.pop(key, None)
        if updated_auto_map:
            merged["auto_map"] = updated_auto_map
        else:
            merged.pop("auto_map", None)
        return
    if key == "AutoTokenizer" and isinstance(merged_auto_map, (list, tuple)):
        merged.pop("auto_map", None)


def _extract_auto_map_value(
    payload: dict | None,
    *,
    key: str,
) -> tuple[object | None, bool, bool]:
    if not isinstance(payload, dict):
        return None, False, False
    auto_map = payload.get("auto_map")
    if key == "AutoTokenizer" and isinstance(auto_map, (list, tuple)):
        if not any(isinstance(value, str) and value for value in auto_map):
            return None, False, False
        return list(auto_map), True, True
    if not isinstance(auto_map, dict) or key not in auto_map:
        return None, False, False
    value = auto_map[key]
    values = value if isinstance(value, (list, tuple)) else [value]
    if not any(isinstance(entry, str) and entry for entry in values):
        return None, False, False
    if isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, list):
        value = list(value)
    return value, True, False


def _collect_remote_code_files(
    model_dir: Path,
    adapter_dir: Path,
    metadata: _OverlayMetadata,
) -> list[Path]:
    queue = deque(
        (_parse_remote_code_module_path(class_ref), 0)
        for class_ref in _collect_remote_code_class_refs(metadata)
    )
    collected: list[Path] = []
    seen: set[Path] = set()
    total_bytes = 0
    while queue:
        relative_path, depth = queue.popleft()
        if relative_path in seen:
            continue
        if depth > _MAX_REMOTE_CODE_DEPTH:
            raise ModelValidationError("Remote-code import graph exceeds the supported depth")
        source_path = _locate_overlay_source(model_dir, adapter_dir, relative_path)
        if source_path is None:
            raise ModelValidationError(f"Remote-code module not found: {relative_path}")
        if len(seen) >= _MAX_REMOTE_CODE_FILES:
            raise ModelValidationError("Remote-code import graph exceeds the supported file budget")
        file_size = source_path.stat().st_size
        if total_bytes + file_size > _MAX_REMOTE_CODE_BYTES:
            raise ModelValidationError("Remote-code import graph exceeds the supported byte budget")
        total_bytes += file_size
        seen.add(relative_path)
        collected.append(relative_path)
        for module_name in _relative_import_module_names(source_path):
            queue.append(
                (
                    _resolve_relative_import_module_path(
                        model_dir,
                        adapter_dir,
                        source_relative_path=relative_path,
                        module_name=module_name,
                    ),
                    depth + 1,
                )
            )
    return collected


def _collect_remote_code_class_refs(metadata: _OverlayMetadata) -> list[str]:
    refs: list[str] = []
    for payload, key in (
        (metadata.tokenizer_config, "AutoTokenizer"),
        (metadata.config, "AutoTokenizer"),
        (metadata.config, "AutoConfig"),
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
        raise ModelValidationError(
            "External remote-code repositories are currently unsupported for local model bundles"
        )
    module_name, _, class_name = class_ref.rpartition(".")
    if not module_name or not class_name:
        raise ModelValidationError(
            f"Remote-code class reference is currently unsupported: {class_ref}"
        )
    if "." in module_name:
        raise ModelValidationError(
            f"Package-scoped remote-code entry points are currently unsupported: {class_ref}"
        )
    return Path(f"{module_name}.py")


def _relative_import_module_names(module_path: Path) -> list[str]:
    try:
        content = module_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelValidationError(f"Could not read Python module from {module_path}") from exc
    try:
        tree = ast.parse(content, filename=str(module_path))
    except SyntaxError as exc:
        raise ModelValidationError(f"Could not parse Python module from {module_path}") from exc
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level > 1:
            raise ModelValidationError(
                f"Parent relative imports in remote-code modules are currently unsupported: {module_path}"
            )
        if node.level != 1:
            continue
        if node.module and "." in node.module:
            raise ModelValidationError(
                f"Package-scoped relative imports in remote-code modules are currently unsupported: {module_path}"
            )
        if node.module:
            names.append(node.module)
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            names.append(alias.name)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _module_name_to_relative_path(module_name: str) -> Path:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        raise ModelValidationError(
            f"Remote-code module reference is currently unsupported: {module_name}"
        )
    return Path(*parts).with_suffix(".py")


def _resolve_relative_import_module_path(
    model_dir: Path,
    adapter_dir: Path,
    *,
    source_relative_path: Path,
    module_name: str,
) -> Path:
    relative_module_path = source_relative_path.parent / _module_name_to_relative_path(module_name)
    package_init_path = source_relative_path.parent / module_name / "__init__.py"
    if _locate_overlay_source(model_dir, adapter_dir, package_init_path) is not None:
        raise ModelValidationError(
            "Package-scoped relative imports in remote-code modules are currently unsupported: "
            f"{source_relative_path}"
        )
    return relative_module_path


def _load_optional_json_with_message(
    path: Path,
    *,
    symlink_message: str,
) -> dict | None:
    _raise_if_symlink(path, symlink_message)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_optional_json_strict(
    path: Path,
    *,
    symlink_message: str,
    transform=None,
) -> dict | None:
    _raise_if_symlink(path, symlink_message)
    if not path.is_file():
        return None
    payload = _load_json(path)
    return payload if transform is None else transform(payload)


def _load_required_json_strict(
    path: Path,
    *,
    symlink_message: str,
    transform=None,
) -> dict:
    _raise_if_symlink(path, symlink_message)
    if not path.is_file():
        raise ModelValidationError(f"{path.name} not found in {path.parent}")
    payload = _load_json(path)
    return payload if transform is None else transform(payload)


def _collect_added_tokens(tokenizer_payload: dict | list | None) -> list[int]:
    if not tokenizer_payload:
        return []
    if isinstance(tokenizer_payload, list):
        raise ModelValidationError("added_tokens metadata is malformed")
    token_ids: list[int] = []
    if isinstance(tokenizer_payload, dict):
        added_tokens = tokenizer_payload.get("added_tokens", [])
        if not isinstance(added_tokens, list):
            raise ModelValidationError("added_tokens metadata is malformed")
        try:
            for token in added_tokens:
                if not isinstance(token, dict):
                    raise ModelValidationError("added_tokens metadata is malformed")
                if token.get("content") in _STOP_TOKEN_NAMES and "id" in token:
                    token_ids.append(int(token["id"]))
            for token_name in _STOP_TOKEN_NAMES:
                token_id = tokenizer_payload.get(token_name)
                if token_id is not None:
                    token_ids.append(int(token_id))
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("added_tokens metadata is malformed") from exc
    return token_ids
