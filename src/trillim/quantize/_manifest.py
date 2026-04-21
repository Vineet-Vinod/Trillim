"""Manifest generation and binary invocation for local quantization."""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
from pathlib import Path

from trillim.components.llm._config import ArchitectureType

from ._config import LORA_TARGETS, ModelQuantizeConfig, layer_index_for_key

ACTION_BF16_RAW = 0
ACTION_TERNARY_QUANTIZE = 1
ACTION_REPACK_TERNARY = 3
ACTION_Q1_0_128 = 4
ACTION_GROUP_TERNARY_QUANTIZE = 5

SECTION_TEXT_CORE = 1

DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_BF16 = 2
DTYPE_I8 = 3
DTYPE_U8 = 4

_SAFETENSORS_DTYPE_MAP = {
    "F32": (DTYPE_F32, 4),
    "F16": (DTYPE_F16, 2),
    "BF16": (DTYPE_BF16, 2),
    "I8": (DTYPE_I8, 1),
    "U8": (DTYPE_U8, 1),
}

_SHORT_TO_FULL = {
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "q_proj": "self_attn.q_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
_QUANTIZE_BINARY_NAME = "trillim-quantize"


def determine_language_model_only(model_dir: Path, config: ModelQuantizeConfig) -> bool:
    if config.arch_type != ArchitectureType.QWEN35:
        return False
    tensor_names = get_all_tensor_names(model_dir)
    return any(
        name.startswith("model.visual.") or name.startswith("mtp.")
        for name in tensor_names
    )


def _is_bonsai_family(arch_type: ArchitectureType) -> bool:
    return arch_type in {ArchitectureType.BONSAI, ArchitectureType.BONSAI_TERNARY}


def _quantized_tensor_action(dtype_str: str, arch_type: ArchitectureType) -> int:
    if arch_type == ArchitectureType.BONSAI:
        return ACTION_Q1_0_128
    if arch_type == ArchitectureType.BONSAI_TERNARY:
        return ACTION_GROUP_TERNARY_QUANTIZE
    if dtype_str in {"I8", "U8"}:
        return ACTION_REPACK_TERNARY
    return ACTION_TERNARY_QUANTIZE


def resolve_quantize_binary() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(__file__).resolve().parents[1] / "_bin" / f"{_QUANTIZE_BINARY_NAME}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled quantizer binary not found at {path}")
    return path


def get_tensor_metadata(input_path: Path) -> list[dict[str, object]]:
    with input_path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    metadata: list[dict[str, object]] = []
    for key, info in header.items():
        if key == "__metadata__":
            continue
        metadata.append(
            {
                "key": key,
                "start": info["data_offsets"][0],
                "shape": info["shape"],
            }
        )
    return metadata


def get_sharded_files(model_dir: Path) -> tuple[list[Path], dict[str, Path]]:
    single_path = model_dir / "model.safetensors"
    if single_path.is_file():
        return [single_path], {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid sharded safetensors index in {index_path}")
        shard_files = sorted({str(value) for value in weight_map.values()})
        shard_paths = [model_dir / shard_name for shard_name in shard_files]
        full_weight_map = {str(name): model_dir / str(filename) for name, filename in weight_map.items()}
        return shard_paths, full_weight_map
    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json found in {model_dir}"
    )


def get_all_tensor_names(model_dir: Path) -> list[str]:
    shard_files, weight_map = get_sharded_files(model_dir)
    if weight_map:
        return list(weight_map.keys())
    return [str(item["key"]) for item in get_tensor_metadata(shard_files[0])]


def validate_adapter_source(adapter_dir: Path, config: ModelQuantizeConfig) -> None:
    _read_adapter_metadata(adapter_dir, config)


def build_manifest(
    model_dir: Path,
    config: ModelQuantizeConfig,
    *,
    output_dir: Path,
    adapter_dir: Path | None = None,
    skip_model: bool = False,
    language_model_only: bool = False,
) -> Path:
    if not skip_model:
        _validate_supported_model_tensors(
            model_dir,
            config,
            language_model_only=language_model_only,
        )
    if skip_model:
        shard_files: list[Path] = []
        weight_map: dict[str, Path] = {}
        has_model_tensors = False
    else:
        try:
            shard_files, weight_map = get_sharded_files(model_dir)
            has_model_tensors = True
        except FileNotFoundError:
            if adapter_dir is None:
                raise
            shard_files, weight_map = [], {}
            has_model_tensors = False

    shard_path_list = list(shard_files)
    shard_idx_map = {path: index for index, path in enumerate(shard_path_list)}
    if weight_map:
        for shard_path in weight_map.values():
            if shard_path not in shard_idx_map:
                shard_idx_map[shard_path] = len(shard_path_list)
                shard_path_list.append(shard_path)

    shard_headers: dict[Path, dict] = {}
    shard_data_starts: dict[Path, int] = {}
    for shard_path in shard_path_list:
        header, data_start = _get_header_and_offsets(shard_path)
        shard_headers[shard_path] = header
        shard_data_starts[shard_path] = data_start

    tensor_entries: list[dict[str, int]] = []
    sections: list[dict[str, int]] = []
    if has_model_tensors:
        if weight_map:
            all_tensors_meta: list[dict[str, object]] = []
            for shard_path in shard_files:
                shard_meta = get_tensor_metadata(shard_path)
                for item in shard_meta:
                    item["file"] = shard_path
                all_tensors_meta.extend(shard_meta)
        else:
            all_tensors_meta = get_tensor_metadata(shard_files[0])
            for item in all_tensors_meta:
                item["file"] = shard_files[0]

        tensors_meta = _ordered_text_tensors(
            all_tensors_meta,
            config,
            language_model_only=language_model_only,
        )
        if tensors_meta:
            sections.append(
                {
                    "type": SECTION_TEXT_CORE,
                    "first_tensor_idx": 0,
                    "num_tensors": len(tensors_meta),
                }
            )
        for item in tensors_meta:
            key = str(item["key"])
            shape = [int(value) for value in item["shape"]]
            file_path = item["file"]

            row = shape[0] if shape else 1
            col = math.prod(shape[1:]) if len(shape) >= 2 else 1

            is_1d = col == 1
            is_embedding = key == config.arch_info.embedding_key
            is_lm_head = key.startswith("lm_head.")
            should_quantize = not (is_1d or is_embedding or is_lm_head)
            if _is_bonsai_family(config.arch_type):
                should_quantize = not is_1d

            padded_shape = list(shape)
            needs_padding = False
            for index, dim_size in enumerate(padded_shape):
                if dim_size == config.intermediate_dim_orig and config.intermediate_dim != config.intermediate_dim_orig:
                    padded_shape[index] = config.intermediate_dim
                    needs_padding = True
                elif dim_size == config.hidden_dim_orig and config.hidden_dim != config.hidden_dim_orig:
                    padded_shape[index] = config.hidden_dim
                    needs_padding = True
            padded_row = padded_shape[0] if padded_shape else 1
            padded_col = math.prod(padded_shape[1:]) if len(padded_shape) >= 2 else 1
            if not needs_padding:
                padded_row = row
                padded_col = col

            header = shard_headers[file_path]
            tensor_info = header[key]
            dtype_str = str(tensor_info["dtype"])
            data_offsets = tensor_info["data_offsets"]
            data_offset_begin = int(data_offsets[0])
            data_offset_end = int(data_offsets[1])
            data_size = data_offset_end - data_offset_begin
            absolute_offset = shard_data_starts[file_path] + data_offset_begin
            dtype_code, _element_size = _safetensors_dtype_code(dtype_str)

            if is_embedding or is_lm_head:
                action = (
                    _quantized_tensor_action(dtype_str, config.arch_type)
                    if _is_bonsai_family(config.arch_type)
                    else ACTION_BF16_RAW
                )
            elif should_quantize:
                action = _quantized_tensor_action(dtype_str, config.arch_type)
            else:
                action = ACTION_BF16_RAW

            has_scale = 0
            scale_shard_idx = 0
            scale_offset = 0
            scale_size = 0
            if action == ACTION_REPACK_TERNARY:
                scale_key = f"{key}_scale"
                scale_file = weight_map.get(scale_key, file_path) if weight_map else file_path
                if scale_file not in shard_idx_map:
                    shard_idx_map[scale_file] = len(shard_path_list)
                    shard_path_list.append(scale_file)
                    header, data_start = _get_header_and_offsets(scale_file)
                    shard_headers[scale_file] = header
                    shard_data_starts[scale_file] = data_start
                scale_header = shard_headers[scale_file]
                if scale_key in scale_header:
                    has_scale = 1
                    scale_info = scale_header[scale_key]
                    scale_offsets = scale_info["data_offsets"]
                    scale_offset = shard_data_starts[scale_file] + int(scale_offsets[0])
                    scale_size = int(scale_offsets[1]) - int(scale_offsets[0])
                    scale_shard_idx = shard_idx_map[scale_file]

            tensor_entries.append(
                {
                    "action": action,
                    "dtype": dtype_code,
                    "row": row,
                    "col": col,
                    "padded_row": padded_row,
                    "padded_col": padded_col,
                    "shard_idx": shard_idx_map[file_path],
                    "data_offset": absolute_offset,
                    "data_size": data_size,
                    "has_scale": has_scale,
                    "scale_shard_idx": scale_shard_idx,
                    "scale_offset": scale_offset,
                    "scale_size": scale_size,
                }
            )

    lora_entries = None
    lora_scale = 0.0
    if adapter_dir is not None:
        lora_entries, lora_scale = _build_lora_entries(
            adapter_dir,
            config,
            shard_path_list,
            shard_idx_map,
            shard_headers,
            shard_data_starts,
        )

    manifest_path = output_dir / ".quantize_manifest.bin"
    with manifest_path.open("wb") as handle:
        handle.write(struct.pack("<H", len(shard_path_list)))
        for shard_path in shard_path_list:
            encoded = str(shard_path).encode("utf-8")
            handle.write(struct.pack("<H", len(encoded)))
            handle.write(encoded)
        handle.write(struct.pack("<I", len(tensor_entries)))
        for entry in tensor_entries:
            handle.write(struct.pack("<B", entry["action"]))
            handle.write(struct.pack("<B", entry["dtype"]))
            handle.write(struct.pack("<I", entry["row"]))
            handle.write(struct.pack("<I", entry["col"]))
            handle.write(struct.pack("<I", entry["padded_row"]))
            handle.write(struct.pack("<I", entry["padded_col"]))
            handle.write(struct.pack("<H", entry["shard_idx"]))
            handle.write(struct.pack("<Q", entry["data_offset"]))
            handle.write(struct.pack("<Q", entry["data_size"]))
            handle.write(struct.pack("<B", entry["has_scale"]))
            handle.write(struct.pack("<H", entry["scale_shard_idx"]))
            handle.write(struct.pack("<Q", entry["scale_offset"]))
            handle.write(struct.pack("<Q", entry["scale_size"]))
        handle.write(struct.pack("<I", len(sections)))
        for section in sections:
            handle.write(struct.pack("<B", section["type"]))
            handle.write(struct.pack("<I", section["first_tensor_idx"]))
            handle.write(struct.pack("<I", section["num_tensors"]))
        if lora_entries is not None:
            handle.write(struct.pack("<I", config.num_layers))
            handle.write(struct.pack("<I", len(LORA_TARGETS)))
            handle.write(struct.pack("<d", lora_scale))
            for layer_targets in lora_entries:
                for target in layer_targets:
                    if target is None:
                        handle.write(struct.pack("<B", 0))
                        continue
                    handle.write(struct.pack("<B", 1))
                    handle.write(struct.pack("<B", target["a_dtype"]))
                    handle.write(struct.pack("<I", target["a_rows"]))
                    handle.write(struct.pack("<I", target["a_cols"]))
                    handle.write(struct.pack("<H", target["a_shard_idx"]))
                    handle.write(struct.pack("<Q", target["a_offset"]))
                    handle.write(struct.pack("<Q", target["a_size"]))
                    handle.write(struct.pack("<B", target["b_dtype"]))
                    handle.write(struct.pack("<I", target["b_rows"]))
                    handle.write(struct.pack("<I", target["b_cols"]))
                    handle.write(struct.pack("<H", target["b_shard_idx"]))
                    handle.write(struct.pack("<Q", target["b_offset"]))
                    handle.write(struct.pack("<Q", target["b_size"]))
    return manifest_path


def run_model_quantizer(
    binary_path: Path,
    model_dir: Path,
    config: ModelQuantizeConfig,
    *,
    output_dir: Path,
    language_model_only: bool,
) -> None:
    manifest_path = build_manifest(
        model_dir,
        config,
        output_dir=output_dir,
        language_model_only=language_model_only,
    )
    command = [
        str(binary_path),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_dir / "qmodel.tensors"),
        "--arch",
        str(int(config.arch_type)),
        "--hidden-dim",
        str(config.hidden_dim),
        "--intermediate-dim",
        str(config.intermediate_dim),
        "--hidden-dim-orig",
        str(config.hidden_dim_orig),
        "--intermediate-dim-orig",
        str(config.intermediate_dim_orig),
        "--norm-eps",
        str(config.norm_eps),
        "--rope-output",
        str(output_dir / "rope.cache"),
        "--rope-theta",
        str(config.rope_theta),
        "--max-pos",
        str(config.max_position_embeddings),
        "--head-dim",
        str(config.head_dim),
        "--rope-dim",
        str(int(round(config.head_dim * config.partial_rotary_factor))),
    ]
    if config.yarn_factor is not None and config.original_max_position_embeddings is not None:
        command.extend(
            [
                "--yarn-factor",
                str(config.yarn_factor),
                "--yarn-orig-max-pos",
                str(config.original_max_position_embeddings),
            ]
        )
    if config.yarn_beta_slow is not None:
        command.extend(["--yarn-beta-slow", str(config.yarn_beta_slow)])
    if config.yarn_beta_fast is not None:
        command.extend(["--yarn-beta-fast", str(config.yarn_beta_fast)])
    if config.tie_word_embeddings:
        command.append("--tie-embeddings")
    try:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"C++ quantizer exited with code {result.returncode}")
    finally:
        _cleanup_paths(manifest_path, output_dir / "qmodel.tensors.tmp")


def run_adapter_quantizer(
    binary_path: Path,
    model_dir: Path,
    config: ModelQuantizeConfig,
    *,
    adapter_dir: Path,
    output_dir: Path,
    language_model_only: bool,
) -> None:
    manifest_path = build_manifest(
        model_dir,
        config,
        output_dir=output_dir,
        adapter_dir=adapter_dir,
        skip_model=True,
        language_model_only=language_model_only,
    )
    adapter_config = _read_adapter_config_file(adapter_dir)
    rank = _require_positive_int(adapter_config.get("r"), "adapter rank")
    temp_output = output_dir / ".unused-qmodel.tensors"
    command = [
        str(binary_path),
        "--manifest",
        str(manifest_path),
        "--output",
        str(temp_output),
        "--arch",
        str(int(config.arch_type)),
        "--hidden-dim",
        str(config.hidden_dim),
        "--intermediate-dim",
        str(config.intermediate_dim),
        "--hidden-dim-orig",
        str(config.hidden_dim_orig),
        "--intermediate-dim-orig",
        str(config.intermediate_dim_orig),
        "--norm-eps",
        str(config.norm_eps),
        "--head-dim",
        str(config.head_dim),
        "--lora-output",
        str(output_dir / "qmodel.lora"),
        "--lora-rank",
        str(rank),
        "--num-heads",
        str(config.num_heads),
        "--num-kv-heads",
        str(config.num_kv_heads),
    ]
    if config.tie_word_embeddings:
        command.append("--tie-embeddings")
    try:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"C++ quantizer exited with code {result.returncode}")
    finally:
        _cleanup_paths(
            manifest_path,
            temp_output,
            output_dir / ".unused-qmodel.tensors.tmp",
        )


def _ordered_text_tensors(
    tensors_meta: list[dict[str, object]],
    config: ModelQuantizeConfig,
    *,
    language_model_only: bool,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for item in tensors_meta:
        key = str(item["key"])
        if _should_skip_tensor(key, tie_word_embeddings=config.tie_word_embeddings):
            continue
        if language_model_only and _is_language_model_only_skip(key):
            continue
        filtered.append(item)
    return sorted(filtered, key=lambda item: _processing_sort_key(str(item["key"]), config))


def _validate_supported_model_tensors(
    model_dir: Path,
    config: ModelQuantizeConfig,
    *,
    language_model_only: bool,
) -> None:
    tensor_names = get_all_tensor_names(model_dir)
    for key in tensor_names:
        if _should_skip_tensor(key, tie_word_embeddings=config.tie_word_embeddings):
            continue
        if _is_language_model_only_skip(key):
            if config.arch_type != ArchitectureType.QWEN35:
                raise ValueError(f"layer unsupported at this time: {key}")
            if not language_model_only:
                raise ValueError(
                    "Qwen3.5 multimodal checkpoints currently support only text-only quantization"
                )
            continue
        if not _is_supported_text_tensor(key, config):
            raise ValueError(f"layer unsupported at this time: {key}")


def _is_supported_text_tensor(key: str, config: ModelQuantizeConfig) -> bool:
    if key == config.arch_info.embedding_key:
        return True
    if key in {config.arch_info.final_norm_key, "lm_head.weight", "lm_head.bias"}:
        return True
    if layer_index_for_key(key, config.arch_info) is None:
        return False
    for component in config.arch_info.component_order:
        if _matches_component_key(key, component):
            return True
    return False


def _matches_component_key(key: str, component: str) -> bool:
    if key.endswith(f".{component}"):
        return True
    if component.endswith((".weight", ".bias")) or component.endswith(("A_log", "dt_bias")):
        return False
    return key.endswith(f".{component}.weight") or key.endswith(f".{component}.bias")


def _processing_sort_key(key: str, config: ModelQuantizeConfig) -> tuple[int, int, int, int]:
    component_order = config.arch_info.component_order
    if key == config.arch_info.embedding_key:
        return (0, -1, -1, 0)
    if key.startswith("lm_head."):
        return (0, 0, -1, 0)
    if key == config.arch_info.final_norm_key:
        return (1, -1, -1, 0)
    layer_index = layer_index_for_key(key, config.arch_info)
    if layer_index is None:
        return (3, -1, -1, 0)
    intra_priority = len(component_order)
    for index, component in enumerate(component_order):
        if _matches_component_key(key, component):
            intra_priority = index
            break
    bias_order = 1 if key.endswith(".bias") else 0
    return (2, layer_index, intra_priority, bias_order)


def _should_skip_tensor(key: str, *, tie_word_embeddings: bool) -> bool:
    if "rotary_emb" in key or "inv_freq" in key:
        return True
    if tie_word_embeddings and key == "lm_head.weight":
        return True
    return key.endswith("_scale")


def _is_language_model_only_skip(key: str) -> bool:
    return key.startswith("model.visual.") or key.startswith("mtp.")


def _get_header_and_offsets(shard_path: Path) -> tuple[dict, int]:
    with shard_path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header_json = handle.read(header_size)
    return json.loads(header_json), 8 + header_size


def _safetensors_dtype_code(dtype_str: str) -> tuple[int, int]:
    try:
        return _SAFETENSORS_DTYPE_MAP[dtype_str]
    except KeyError as exc:
        raise ValueError(f"Unknown safetensors dtype: {dtype_str}") from exc


def _build_lora_entries(
    adapter_dir: Path,
    config: ModelQuantizeConfig,
    shard_path_list: list[Path],
    shard_idx_map: dict[Path, int],
    shard_headers: dict[Path, dict],
    shard_data_starts: dict[Path, int],
) -> tuple[list[list[dict[str, int] | None]], float]:
    adapter_config = _read_adapter_metadata(adapter_dir, config)
    rank = _require_positive_int(adapter_config.get("r"), "adapter rank")
    alpha = adapter_config.get("lora_alpha", rank)
    use_rslora = bool(adapter_config.get("use_rslora", False))
    scale = float(alpha) / ((rank**0.5) if use_rslora else rank)

    adapter_path = adapter_dir / "adapter_model.safetensors"
    if adapter_path not in shard_idx_map:
        shard_idx_map[adapter_path] = len(shard_path_list)
        shard_path_list.append(adapter_path)
        header, data_start = _get_header_and_offsets(adapter_path)
        shard_headers[adapter_path] = header
        shard_data_starts[adapter_path] = data_start

    adapter_header = shard_headers[adapter_path]
    adapter_shard_idx = shard_idx_map[adapter_path]
    adapter_data_start = shard_data_starts[adapter_path]

    entries: list[list[dict[str, int] | None]] = []
    for layer_index in range(config.num_layers):
        layer_targets: list[dict[str, int] | None] = []
        for target in LORA_TARGETS:
            key_a = _find_lora_key(adapter_header, layer_index, target, "A")
            key_b = _find_lora_key(adapter_header, layer_index, target, "B")
            if key_a is None or key_b is None:
                layer_targets.append(None)
                continue
            a_info = adapter_header[key_a]
            b_info = adapter_header[key_b]
            a_dtype, _ = _safetensors_dtype_code(a_info["dtype"])
            b_dtype, _ = _safetensors_dtype_code(b_info["dtype"])
            a_offsets = a_info["data_offsets"]
            b_offsets = b_info["data_offsets"]
            layer_targets.append(
                {
                    "a_dtype": a_dtype,
                    "a_rows": int(a_info["shape"][0]),
                    "a_cols": int(a_info["shape"][1]),
                    "a_shard_idx": adapter_shard_idx,
                    "a_offset": adapter_data_start + int(a_offsets[0]),
                    "a_size": int(a_offsets[1]) - int(a_offsets[0]),
                    "b_dtype": b_dtype,
                    "b_rows": int(b_info["shape"][0]),
                    "b_cols": int(b_info["shape"][1]),
                    "b_shard_idx": adapter_shard_idx,
                    "b_offset": adapter_data_start + int(b_offsets[0]),
                    "b_size": int(b_offsets[1]) - int(b_offsets[0]),
                }
            )
        entries.append(layer_targets)
    return entries, scale


def _read_adapter_metadata(adapter_dir: Path, config: ModelQuantizeConfig) -> dict:
    adapter_config = _read_adapter_config_file(adapter_dir)
    rank = _require_positive_int(adapter_config.get("r"), "adapter rank")
    raw_targets = adapter_config.get("target_modules", [])
    if not isinstance(raw_targets, list):
        raise ValueError("target_modules must be a list")
    for raw_target in raw_targets:
        normalized = _SHORT_TO_FULL.get(str(raw_target), str(raw_target))
        if normalized not in LORA_TARGETS:
            raise ValueError(f"layer unsupported at this time: {raw_target}")

    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"{adapter_path} not found")
    header, _data_start = _get_header_and_offsets(adapter_path)
    max_layer_index = -1
    for key, info in header.items():
        if key == "__metadata__":
            continue
        parsed = _parse_adapter_tensor_key(key)
        if parsed is None:
            raise ValueError(f"layer unsupported at this time: {key}")
        layer_index, target, part = parsed
        if target not in LORA_TARGETS:
            raise ValueError(f"layer unsupported at this time: {key}")
        max_layer_index = max(max_layer_index, layer_index)
        shape = [int(value) for value in info["shape"]]
        expected_input_dim, expected_output_dim = _expected_lora_dims(config, target)
        if part == "A":
            if len(shape) != 2 or shape[0] != rank:
                raise ValueError(
                    f"Adapter tensor {key} has rank {shape[0] if shape else 'unknown'} on lora_A but adapter_config.json declares r={rank}."
                )
            if shape[1] != expected_input_dim:
                raise ValueError(
                    f"Adapter {target} lora_A on layer {layer_index} has input dim {shape[1]} but the base model expects {expected_input_dim}."
                )
        else:
            if len(shape) != 2 or shape[1] != rank:
                raise ValueError(
                    f"Adapter tensor {key} has rank {shape[1] if len(shape) > 1 else 'unknown'} on lora_B but adapter_config.json declares r={rank}."
                )
            if shape[0] != expected_output_dim:
                raise ValueError(
                    f"Adapter {target} lora_B on layer {layer_index} has output dim {shape[0]} but the base model expects {expected_output_dim}."
                )
    if max_layer_index + 1 > config.num_layers:
        raise ValueError(
            f"Adapter has weights for {max_layer_index + 1} layers but the base model only has {config.num_layers} layers."
        )
    return adapter_config


def _read_adapter_config_file(adapter_dir: Path) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _find_lora_key(header: dict, layer_index: int, target: str, part: str) -> str | None:
    candidates = (
        f"base_model.model.model.layers.{layer_index}.{target}.lora_{part}.weight",
        f"model.layers.{layer_index}.{target}.lora_{part}.weight",
    )
    for key in candidates:
        if key in header:
            return key
    return None


def _parse_adapter_tensor_key(key: str) -> tuple[int, str, str] | None:
    prefixes = (
        "base_model.model.model.layers.",
        "model.layers.",
    )
    for prefix in prefixes:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        for target in LORA_TARGETS:
            for part in ("A", "B"):
                needle = f".{target}.lora_{part}.weight"
                if not suffix.endswith(needle):
                    continue
                layer_text = suffix[: -len(needle)]
                if layer_text.isdigit():
                    return int(layer_text), target, part
        return None
    return None


def _expected_lora_dims(
    config: ModelQuantizeConfig,
    target: str,
) -> tuple[int, int]:
    attention_output_dim = config.num_heads * config.head_dim
    key_value_output_dim = config.num_kv_heads * config.head_dim
    if target == "self_attn.q_proj":
        return config.hidden_dim_orig, attention_output_dim
    if target in {"self_attn.k_proj", "self_attn.v_proj"}:
        return config.hidden_dim_orig, key_value_output_dim
    if target == "self_attn.o_proj":
        return attention_output_dim, config.hidden_dim_orig
    if target in {"mlp.gate_proj", "mlp.up_proj"}:
        return config.hidden_dim_orig, config.intermediate_dim_orig
    if target == "mlp.down_proj":
        return config.intermediate_dim_orig, config.hidden_dim_orig
    raise ValueError(f"layer unsupported at this time: {target}")


def _require_positive_int(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


def _cleanup_paths(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
