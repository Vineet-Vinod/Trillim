# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""
Standalone quantization command and binary manifest writer.

Quantizes safetensors weights and/or extracts LoRA adapters for use with
the DarkNet inference engine. All heavy lifting is done by the proprietary
C++ quantizer binary.

Also provides safetensor I/O utilities (shard discovery, header parsing, tensor
ordering) and the binary manifest builder used by the C++ quantizer.

Usage:
    trillim quantize <model_dir> --model                          # → <model_dir>-TRNQ/
    trillim quantize <model_dir> --adapter <adapter_dir>          # → <adapter_dir>-TRNQ/
    trillim quantize <model_dir> --model --adapter <adapter_dir>  # both -TRNQ/ dirs
"""

import json
import os
import re
import shutil
import struct
import subprocess
import tempfile

from trillim.model_arch import LORA_TARGETS, ModelConfig
from trillim.utils import compute_base_model_hash


# ---------------------------------------------------------------------------
# Safetensor utilities
# ---------------------------------------------------------------------------

def get_tensor_metadata(input_path):
    """Parse Safetensors header for tensor metadata."""
    with open(input_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json_bytes = f.read(header_size)
        header = json.loads(header_json_bytes)

    tensors_meta = []
    for key, info in header.items():
        if key == "__metadata__":
            continue
        tensors_meta.append(
            {
                "key": key,
                "start": info["data_offsets"][0],
                "shape": info["shape"],
            }
        )

    return tensors_meta


def get_sharded_files(model_dir):
    """
    Get list of safetensor files and weight map for sharded models.
    Returns (list of file paths, dict mapping tensor name -> file path).
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(single_path):
        return [single_path], {}

    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        shard_paths = [os.path.join(model_dir, f) for f in shard_files]
        full_weight_map = {k: os.path.join(model_dir, v) for k, v in weight_map.items()}

        return shard_paths, full_weight_map

    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json found in {model_dir}"
    )


def get_all_tensor_names(model_dir):
    """Get all tensor names from a model (handles sharding)."""
    shard_files, weight_map = get_sharded_files(model_dir)

    if weight_map:
        return list(weight_map.keys())
    else:
        tensors_meta = get_tensor_metadata(shard_files[0])
        return [t["key"] for t in tensors_meta]


def get_processing_order(tensors_meta, arch_info):
    """Sort tensors based on architecture-specific component order."""
    component_order = arch_info.component_order

    def get_intra_layer_priority(key):
        for idx, sub in enumerate(component_order):
            if sub in key:
                return idx
        return 999

    def sort_key(item):
        key = item["key"]

        if arch_info.embedding_pattern in key and "lm_head" not in key:
            return (0, -1, -1, 0)

        if "lm_head" in key:
            return (0, 0, -1, 0)

        if key == arch_info.final_norm_pattern:
            return (1, -1, -1, 0)

        match = re.search(arch_info.layer_pattern, key)
        if match:
            layer_idx = int(match.group(1))
            intra_priority = get_intra_layer_priority(key)
            bias_order = 1 if key.endswith(".bias") else 0
            return (2, layer_idx, intra_priority, bias_order)

        return (3, -1, -1, 0)

    return sorted(tensors_meta, key=sort_key)


# ---------------------------------------------------------------------------
# Manifest constants
# ---------------------------------------------------------------------------

# Action codes (match C++ TensorAction enum)
ACTION_BF16_RAW         = 0
ACTION_TERNARY_QUANTIZE = 1
# ACTION_EMBEDDING_I8     = 2 (removed in Quantization v2 update)
ACTION_REPACK_TERNARY   = 3

# Dtype codes (match C++ DType enum)
DTYPE_F32  = 0
DTYPE_F16  = 1
DTYPE_BF16 = 2
DTYPE_I8   = 3
DTYPE_U8   = 4

# Safetensors dtype string -> (code, element_size)
_SAFETENSORS_DTYPE_MAP = {
    "F32":  (DTYPE_F32,  4),
    "F16":  (DTYPE_F16,  2),
    "BF16": (DTYPE_BF16, 2),
    "I8":   (DTYPE_I8,   1),
    "U8":   (DTYPE_U8,   1),
}


# ---------------------------------------------------------------------------
# Manifest builder internals
# ---------------------------------------------------------------------------

def _get_header_and_offsets(shard_path):
    """Parse safetensors header, return (header_dict, data_start_offset)."""
    with open(shard_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size)
    header = json.loads(header_json)
    data_start = 8 + header_size
    return header, data_start


def _safetensors_dtype_code(dtype_str):
    """Convert safetensors dtype string to (code, element_size)."""
    entry = _SAFETENSORS_DTYPE_MAP.get(dtype_str)
    if entry is None:
        raise ValueError(f"Unknown safetensors dtype: {dtype_str}")
    return entry


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def write_manifest(model_dir, config: ModelConfig, adapter_dir=None, skip_model=False,
                    manifest_dir=None):
    """Write a binary manifest for the C++ quantizer.

    Returns the path to the written manifest file.

    When skip_model is True or no safetensors exist (adapter-only mode),
    the manifest contains 0 tensor entries and only LoRA entries.

    manifest_dir overrides where the temp manifest is written (defaults to
    model_dir).  Use this to avoid writing into a read-only model directory.
    """
    # Try to load base model safetensors; allow adapter-only mode
    if skip_model:
        shard_files, weight_map = [], {}
        has_model_tensors = False
    else:
        try:
            shard_files, weight_map = get_sharded_files(model_dir)
            has_model_tensors = True
        except FileNotFoundError:
            if not adapter_dir:
                raise
            shard_files, weight_map = [], {}
            has_model_tensors = False

    is_sharded = len(weight_map) > 0

    shard_path_list = list(shard_files)
    shard_idx_map = {p: i for i, p in enumerate(shard_path_list)}

    if is_sharded:
        for tensor_name, shard_path in weight_map.items():
            if shard_path not in shard_idx_map:
                shard_idx_map[shard_path] = len(shard_path_list)
                shard_path_list.append(shard_path)

    # Parse all shard headers
    shard_headers = {}
    shard_data_starts = {}
    for shard_path in shard_path_list:
        header, data_start = _get_header_and_offsets(shard_path)
        shard_headers[shard_path] = header
        shard_data_starts[shard_path] = data_start

    # Build tensor entries (skip if no base model safetensors)
    tensor_entries = []
    if has_model_tensors:
        # Get all tensor metadata
        if is_sharded:
            all_tensors_meta = []
            for shard_path in shard_files:
                shard_meta = get_tensor_metadata(shard_path)
                for t in shard_meta:
                    t["file"] = shard_path
                all_tensors_meta.extend(shard_meta)
        else:
            all_tensors_meta = get_tensor_metadata(shard_files[0])
            for t in all_tensors_meta:
                t["file"] = shard_files[0]

        tie_word_embeddings = config.tie_word_embeddings

        # Filter tensors
        filtered_meta = [
            t for t in all_tensors_meta
            if "rotary_emb" not in t["key"] and "inv_freq" not in t["key"]
        ]
        if tie_word_embeddings:
            filtered_meta = [t for t in filtered_meta if "lm_head" not in t["key"]]
        filtered_meta = [t for t in filtered_meta if not t["key"].endswith("_scale")]

        tensors_meta = get_processing_order(filtered_meta, config.arch_info)

        for item in tensors_meta:
            key = item["key"]
            shape = item["shape"]
            file_path = item["file"]

            row = shape[0] if len(shape) >= 1 else 1
            col = shape[1] if len(shape) >= 2 else 1

            is_1d = col == 1
            is_embedding = (config.arch_info.embedding_pattern in key and "lm_head" not in key)
            is_lm_head = "lm_head" in key
            should_quantize = not (is_1d or is_embedding or is_lm_head)

            # Padding detection
            padded_row = row
            padded_col = col
            orig_shape = list(shape)
            needs_padding = False
            for dim_idx in range(len(orig_shape)):
                dim_size = orig_shape[dim_idx]
                if (dim_size == config.intermediate_dim_orig
                        and config.intermediate_dim != config.intermediate_dim_orig):
                    orig_shape[dim_idx] = config.intermediate_dim
                    needs_padding = True
                elif (dim_size == config.hidden_dim_orig
                      and config.hidden_dim != config.hidden_dim_orig):
                    orig_shape[dim_idx] = config.hidden_dim
                    needs_padding = True

            if needs_padding:
                padded_row = orig_shape[0] if len(orig_shape) >= 1 else 1
                padded_col = orig_shape[1] if len(orig_shape) >= 2 else 1

            # Look up absolute offset and dtype
            header = shard_headers[file_path]
            tensor_info = header[key]
            dtype_str = tensor_info["dtype"]
            data_offsets = tensor_info["data_offsets"]
            data_offset_begin = data_offsets[0]
            data_offset_end = data_offsets[1]
            data_size = data_offset_end - data_offset_begin
            absolute_offset = shard_data_starts[file_path] + data_offset_begin

            dtype_code, _ = _safetensors_dtype_code(dtype_str)

            # Determine action
            if is_embedding or is_lm_head:
                action = ACTION_BF16_RAW
            elif should_quantize:
                action = ACTION_REPACK_TERNARY if dtype_str in ("I8", "U8") else ACTION_TERNARY_QUANTIZE
            else:
                action = ACTION_BF16_RAW

            # Scale tensor info (for REPACK_TERNARY)
            has_scale = 0
            scale_shard_idx = 0
            scale_offset = 0
            scale_size = 0

            if action == ACTION_REPACK_TERNARY:
                scale_key = key + "_scale"
                scale_file = weight_map.get(scale_key, file_path) if is_sharded else file_path

                if scale_file not in shard_idx_map:
                    shard_idx_map[scale_file] = len(shard_path_list)
                    shard_path_list.append(scale_file)
                    h, ds = _get_header_and_offsets(scale_file)
                    shard_headers[scale_file] = h
                    shard_data_starts[scale_file] = ds

                scale_header = shard_headers[scale_file]
                if scale_key in scale_header:
                    has_scale = 1
                    scale_info = scale_header[scale_key]
                    s_offsets = scale_info["data_offsets"]
                    scale_offset = shard_data_starts[scale_file] + s_offsets[0]
                    scale_size = s_offsets[1] - s_offsets[0]
                    scale_shard_idx = shard_idx_map[scale_file]

            shard_idx = shard_idx_map[file_path]

            tensor_entries.append({
                "action": action,
                "dtype": dtype_code,
                "row": row,
                "col": col,
                "padded_row": padded_row,
                "padded_col": padded_col,
                "shard_idx": shard_idx,
                "data_offset": absolute_offset,
                "data_size": data_size,
                "has_scale": has_scale,
                "scale_shard_idx": scale_shard_idx,
                "scale_offset": scale_offset,
                "scale_size": scale_size,
            })

    # LoRA section
    lora_entries = None
    lora_scale = 0.0
    if adapter_dir:
        lora_entries, lora_scale = _build_lora_entries(
            adapter_dir, config, shard_path_list, shard_idx_map,
            shard_headers, shard_data_starts
        )

    # Write binary manifest
    manifest_path = os.path.join(manifest_dir or model_dir, ".quantize_manifest.bin")
    with open(manifest_path, "wb") as f:
        # Shard paths table
        f.write(struct.pack("<H", len(shard_path_list)))
        for p in shard_path_list:
            p_bytes = p.encode("utf-8")
            f.write(struct.pack("<H", len(p_bytes)))
            f.write(p_bytes)

        # Tensor entries
        f.write(struct.pack("<I", len(tensor_entries)))
        for e in tensor_entries:
            f.write(struct.pack("<B", e["action"]))
            f.write(struct.pack("<B", e["dtype"]))
            f.write(struct.pack("<I", e["row"]))
            f.write(struct.pack("<I", e["col"]))
            f.write(struct.pack("<I", e["padded_row"]))
            f.write(struct.pack("<I", e["padded_col"]))
            f.write(struct.pack("<H", e["shard_idx"]))
            f.write(struct.pack("<Q", e["data_offset"]))
            f.write(struct.pack("<Q", e["data_size"]))
            f.write(struct.pack("<B", e["has_scale"]))
            f.write(struct.pack("<H", e["scale_shard_idx"]))
            f.write(struct.pack("<Q", e["scale_offset"]))
            f.write(struct.pack("<Q", e["scale_size"]))

        # LoRA entries
        if lora_entries is not None:
            num_layers = config.num_layers
            targets_per_layer = len(LORA_TARGETS)
            f.write(struct.pack("<I", num_layers))
            f.write(struct.pack("<I", targets_per_layer))
            f.write(struct.pack("<d", lora_scale))

            for layer_targets in lora_entries:
                for lt in layer_targets:
                    if lt is None:
                        f.write(struct.pack("<B", 0))
                    else:
                        f.write(struct.pack("<B", 1))
                        f.write(struct.pack("<B", lt["a_dtype"]))
                        f.write(struct.pack("<I", lt["a_rows"]))
                        f.write(struct.pack("<I", lt["a_cols"]))
                        f.write(struct.pack("<H", lt["a_shard_idx"]))
                        f.write(struct.pack("<Q", lt["a_offset"]))
                        f.write(struct.pack("<Q", lt["a_size"]))
                        f.write(struct.pack("<B", lt["b_dtype"]))
                        f.write(struct.pack("<I", lt["b_rows"]))
                        f.write(struct.pack("<I", lt["b_cols"]))
                        f.write(struct.pack("<H", lt["b_shard_idx"]))
                        f.write(struct.pack("<Q", lt["b_offset"]))
                        f.write(struct.pack("<Q", lt["b_size"]))

    return manifest_path


_SHORT_TO_FULL = {
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "q_proj": "self_attn.q_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


def _build_lora_entries(adapter_dir, config, shard_path_list, shard_idx_map,
                        shard_headers, shard_data_starts):
    """Build LoRA manifest entries from adapter directory."""
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_config_path}"
        )
    with open(adapter_config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)

    rank = adapter_config["r"]
    alpha = adapter_config.get("lora_alpha", rank)
    use_rslora = adapter_config.get("use_rslora", False)

    if use_rslora:
        scale = alpha / (rank ** 0.5)
    else:
        scale = alpha / rank

    raw_targets = adapter_config.get("target_modules", [])
    target_modules = [_SHORT_TO_FULL.get(t, t) for t in raw_targets]
    supported = set(LORA_TARGETS)
    target_modules = [t for t in target_modules if t in supported]

    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if adapter_path not in shard_idx_map:
        shard_idx_map[adapter_path] = len(shard_path_list)
        shard_path_list.append(adapter_path)
        header, data_start = _get_header_and_offsets(adapter_path)
        shard_headers[adapter_path] = header
        shard_data_starts[adapter_path] = data_start

    adapter_header = shard_headers[adapter_path]
    adapter_shard_idx = shard_idx_map[adapter_path]
    adapter_data_start = shard_data_starts[adapter_path]
    adapter_keys = set(k for k in adapter_header.keys() if k != "__metadata__")

    def _find_lora_key(layer_idx, target, ab):
        candidates = [
            f"base_model.model.model.layers.{layer_idx}.{target}.lora_{ab}.weight",
            f"model.layers.{layer_idx}.{target}.lora_{ab}.weight",
        ]
        for key in candidates:
            if key in adapter_keys:
                return key
        return None

    num_layers = config.num_layers
    lora_entries = []

    for layer_idx in range(num_layers):
        layer_targets = []
        for target in LORA_TARGETS:
            key_a = _find_lora_key(layer_idx, target, "A")
            key_b = _find_lora_key(layer_idx, target, "B")

            if key_a is not None and key_b is not None:
                a_info = adapter_header[key_a]
                b_info = adapter_header[key_b]

                a_dtype_code, _ = _safetensors_dtype_code(a_info["dtype"])
                b_dtype_code, _ = _safetensors_dtype_code(b_info["dtype"])

                a_offsets = a_info["data_offsets"]
                b_offsets = b_info["data_offsets"]

                layer_targets.append({
                    "a_dtype": a_dtype_code,
                    "a_rows": a_info["shape"][0],
                    "a_cols": a_info["shape"][1],
                    "a_shard_idx": adapter_shard_idx,
                    "a_offset": adapter_data_start + a_offsets[0],
                    "a_size": a_offsets[1] - a_offsets[0],
                    "b_dtype": b_dtype_code,
                    "b_rows": b_info["shape"][0],
                    "b_cols": b_info["shape"][1],
                    "b_shard_idx": adapter_shard_idx,
                    "b_offset": adapter_data_start + b_offsets[0],
                    "b_size": b_offsets[1] - b_offsets[0],
                })
            else:
                layer_targets.append(None)

        lora_entries.append(layer_targets)

    return lora_entries, scale


# ---------------------------------------------------------------------------
# Quantization command
# ---------------------------------------------------------------------------

def _find_quantize_binary():
    """Find the trillim-quantize binary."""
    from trillim._bin_path import quantize_bin

    return quantize_bin()


def _run_cpp_quantizer(binary_path, model_dir, config, model_output_dir, adapter_dir=None,
                       adapter_output_dir=None):
    """Invoke the C++ quantizer to produce qmodel.tensors, rope.cache, and/or qmodel.lora."""
    if model_output_dir and os.path.realpath(model_output_dir) == os.path.realpath(model_dir):
        raise ValueError("model_output_dir and model_dir resolve to the same path.")

    print("  Writing binary manifest...")
    manifest_path = write_manifest(
        model_dir, config, adapter_dir=adapter_dir,
        manifest_dir=model_output_dir,
    )
    print(f"  Manifest: {manifest_path}")

    cmd = [
        binary_path,
        "--manifest", manifest_path,
        "--output", os.path.join(model_output_dir, "qmodel.tensors"),
        "--arch", str(int(config.arch_type)),
        "--hidden-dim", str(config.hidden_dim),
        "--intermediate-dim", str(config.intermediate_dim),
        "--hidden-dim-orig", str(config.hidden_dim_orig),
        "--intermediate-dim-orig", str(config.intermediate_dim_orig),
        "--norm-eps", str(config.norm_eps),
    ]

    if config.tie_word_embeddings:
        cmd.append("--tie-embeddings")

    # RoPE args
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        raw_config = json.load(f)
    rope_theta = raw_config.get("rope_theta", 500000.0)
    max_pos = raw_config.get("max_position_embeddings", 4096)

    cmd += [
        "--rope-output", os.path.join(model_output_dir, "rope.cache"),
        "--rope-theta", str(rope_theta),
        "--max-pos", str(max_pos),
        "--head-dim", str(config.head_dim),
    ]

    # LoRA args
    if adapter_dir:
        if not adapter_output_dir:
            raise ValueError("adapter_output_dir is required when adapter_dir is set")
        if os.path.realpath(adapter_output_dir) == os.path.realpath(adapter_dir):
            raise ValueError("adapter_output_dir and adapter_dir resolve to the same path.")
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_config = json.load(f)
        rank = adapter_config["r"]

        cmd += [
            "--lora-output", os.path.join(adapter_output_dir, "qmodel.lora"),
            "--lora-rank", str(rank),
            "--num-heads", str(config.num_heads),
            "--num-kv-heads", str(config.num_kv_heads),
        ]

    print(f"  Running: {os.path.basename(binary_path)}")
    try:
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"C++ quantizer exited with code {result.returncode}")
    finally:
        # Clean up manifest and any leftover .tmp from a killed quantizer
        output_tmp = os.path.join(model_output_dir, "qmodel.tensors.tmp")
        for path in (manifest_path, output_tmp):
            if os.path.exists(path):
                os.remove(path)


def _run_cpp_lora_only(binary_path, model_dir, config, adapter_dir,
                       adapter_output_dir):
    """Invoke the C++ quantizer for LoRA-only extraction (no model tensors)."""
    if os.path.realpath(adapter_output_dir) == os.path.realpath(adapter_dir):
        raise ValueError("adapter_output_dir and adapter_dir resolve to the same path.")

    print("  Writing LoRA manifest...")
    manifest_path = write_manifest(model_dir, config, adapter_dir=adapter_dir, skip_model=True,
                                   manifest_dir=adapter_output_dir)
    print(f"  Manifest: {manifest_path}")

    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(adapter_config_path, encoding="utf-8") as f:
        adapter_config = json.load(f)
    rank = adapter_config["r"]

    # Use a temp file for --output since the binary requires it,
    # but we don't need qmodel.tensors in LoRA-only mode
    with tempfile.NamedTemporaryFile(suffix=".tensors", delete=False) as tmp:
        tmp_output = tmp.name

    try:
        cmd = [
            binary_path,
            "--manifest", manifest_path,
            "--output", tmp_output,
            "--arch", str(int(config.arch_type)),
            "--hidden-dim", str(config.hidden_dim),
            "--intermediate-dim", str(config.intermediate_dim),
            "--hidden-dim-orig", str(config.hidden_dim_orig),
            "--intermediate-dim-orig", str(config.intermediate_dim_orig),
            "--norm-eps", str(config.norm_eps),
            "--head-dim", str(config.head_dim),
            "--lora-output", os.path.join(adapter_output_dir, "qmodel.lora"),
            "--lora-rank", str(rank),
            "--num-heads", str(config.num_heads),
            "--num-kv-heads", str(config.num_kv_heads),
        ]

        if config.tie_word_embeddings:
            cmd.append("--tie-embeddings")

        print(f"  Running: {os.path.basename(binary_path)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"C++ quantizer exited with code {result.returncode}")
    finally:
        # Clean up temp files
        for path in (manifest_path, tmp_output, tmp_output + ".tmp"):
            if os.path.exists(path):
                os.remove(path)


def _copy_adapter_tokenizer_files(adapter_dir, output_dir):
    """Copy tokenizer override files from adapter directory to output directory."""
    adapter_tokenizer_config = os.path.join(adapter_dir, "tokenizer_config.json")
    if os.path.exists(adapter_tokenizer_config):
        with open(adapter_tokenizer_config, encoding="utf-8") as f:
            adapter_tok_cfg = json.load(f)
        override_fields = [
            "chat_template",
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
        ]
        overrides = {
            k: adapter_tok_cfg[k] for k in override_fields if k in adapter_tok_cfg
        }
        if overrides:
            override_path = os.path.join(output_dir, "lora_tokenizer_config.json")
            with open(override_path, "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2)
            print(f"  Saved tokenizer overrides: {override_path}")
            for k, v in overrides.items():
                val_str = repr(v) if len(repr(v)) < 40 else repr(v)[:37] + "..."
                print(f"    {k}: {val_str}")

    adapter_chat_template = os.path.join(adapter_dir, "chat_template.jinja")
    if os.path.exists(adapter_chat_template):
        dest_path = os.path.join(output_dir, "lora_chat_template.jinja")
        shutil.copy(adapter_chat_template, dest_path)
        print(f"  Copied chat template: {dest_path}")

    adapter_tokenizer = os.path.join(adapter_dir, "tokenizer.json")
    if os.path.exists(adapter_tokenizer):
        dest_path = os.path.join(output_dir, "lora_tokenizer.json")
        shutil.copy(adapter_tokenizer, dest_path)
        print(f"  Copied tokenizer: {dest_path}")


def _make_adapter_output_dir(adapter_dir):
    """Create a -TRNQ output directory for the quantized adapter.

    Takes the adapter directory name, appends -TRNQ, and creates it as a
    sibling directory.  Returns the absolute path.
    """
    adapter_dir = os.path.abspath(adapter_dir)
    base = adapter_dir.rstrip("/")
    output_dir = base + "-TRNQ"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _make_model_output_dir(model_dir):
    """Create a -TRNQ output directory for the quantized model.

    Takes the model directory name, appends -TRNQ, and creates it as a
    sibling directory.  Returns the absolute path.
    """
    model_dir = os.path.abspath(model_dir)
    base = model_dir.rstrip("/")
    output_dir = base + "-TRNQ"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _copy_model_files(model_dir, output_dir):
    """Copy non-weight files from model_dir to the TRNQ output directory.

    Copies everything needed for inference (config.json, tokenizer files,
    generation_config.json, chat templates, custom tokenizer code, etc.)
    while skipping large weight files and quantization artifacts that are
    generated fresh.
    """
    skip_patterns = [
        "qmodel",
        "rope.cache",
        ".quantize_manifest.bin",
        "safetensors",
    ]

    for entry in os.listdir(model_dir):
        if any(pattern in entry for pattern in skip_patterns):
            continue

        src = os.path.join(model_dir, entry)
        dst = os.path.join(output_dir, entry)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def _validate_adapter_dims(adapter_dir, config):
    """Check that LoRA tensor dimensions are compatible with the model config.

    Reads the adapter safetensors header (no weight data) and verifies that
    representative LoRA matrix shapes match the model's hidden/intermediate
    dimensions and layer count.  Raises ``SystemExit`` on mismatch so the
    user gets a clear error before any heavy quantization work starts.
    """
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        return  # Missing file is caught later by _build_lora_entries

    header, _ = _get_header_and_offsets(st_path)
    tensor_names = [k for k in header if k != "__metadata__"]

    # Detect the key prefix used in this adapter (PEFT vs plain)
    prefixes = ["base_model.model.model.layers.", "model.layers."]

    def _find(layer, target, ab):
        for pfx in prefixes:
            key = f"{pfx}{layer}.{target}.lora_{ab}.weight"
            if key in header:
                return header[key]
        return None

    # --- Check layer count ---
    max_layer = -1
    layer_re = re.compile(r"layers\.(\d+)\.")
    for name in tensor_names:
        m = layer_re.search(name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))

    if max_layer >= 0 and (max_layer + 1) > config.num_layers:
        raise ValueError(
            f"Adapter has weights for {max_layer + 1} layers but the "
            f"base model only has {config.num_layers} layers. "
            "This adapter was not trained on this model."
        )

    # --- Check hidden dimension via q_proj lora_A (shape [rank, hidden_size]) ---
    expected_hidden = config.hidden_dim_orig or config.hidden_dim
    a_info = _find(0, "self_attn.q_proj", "A")
    if a_info is not None:
        cols = a_info["shape"][1]  # [rank, in_features]
        if cols != expected_hidden:
            raise ValueError(
                f"Adapter q_proj lora_A has input dim {cols} but the "
                f"base model hidden_size is {expected_hidden}. "
                "This adapter was not trained on this model."
            )

    # --- Check intermediate dimension via gate_proj lora_B
    #     (shape [intermediate_size, rank]) ---
    expected_intermediate = config.intermediate_dim_orig or config.intermediate_dim
    b_info = _find(0, "mlp.gate_proj", "B")
    if b_info is not None:
        rows = b_info["shape"][0]  # [out_features, rank]
        if rows != expected_intermediate:
            raise ValueError(
                f"Adapter gate_proj lora_B has output dim {rows} but "
                f"the base model intermediate_size is {expected_intermediate}. "
                "This adapter was not trained on this model."
            )


def _write_trillim_model_config(output_dir, config, model_dir):
    """Write trillim_config.json into the quantized model output directory."""
    # Try to read source_model from the model's config.json
    source_model = ""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
        # Use _name_or_path if available (set by HuggingFace on download)
        source_model = raw.get("_name_or_path", "")

    trillim_cfg = {
        "trillim_version": "0.2.0",
        "format_version": 2,
        "type": "model",
        "quantization": "ternary",
        "source_model": source_model,
        "architecture": config.arch_type.name.lower(),
        "platforms": ["x86_64", "aarch64"],
        "base_model_config_hash": compute_base_model_hash(model_dir),
    }

    cfg_path = os.path.join(output_dir, "trillim_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(trillim_cfg, f, indent=2)
        f.write("\n")
    print(f"  Written: {cfg_path}")


def _write_trillim_adapter_config(output_dir, config, adapter_dir, model_dir):
    """Write trillim_config.json into the quantized adapter output directory."""
    # Try to read source_model from the base model's config.json
    source_model = ""
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, encoding="utf-8") as f:
            acfg = json.load(f)
        source_model = acfg.get("base_model_name_or_path", "")

    trillim_cfg = {
        "trillim_version": "0.2.0",
        "format_version": 2,
        "type": "lora_adapter",
        "quantization": "ternary",
        "source_model": source_model,
        "architecture": config.arch_type.name.lower(),
        "platforms": ["x86_64", "aarch64"],
        "base_model_config_hash": compute_base_model_hash(model_dir),
    }

    cfg_path = os.path.join(output_dir, "trillim_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(trillim_cfg, f, indent=2)
        f.write("\n")
    print(f"  Written: {cfg_path}")


def main():
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(
        prog="trillim quantize",
        description="Quantize safetensors weights and/or extract LoRA adapters",
    )
    parser.add_argument("model_dir", help="Path to model directory with config.json")
    parser.add_argument(
        "--model", action="store_true",
        help="Quantize model weights → <model_dir>-TRNQ/",
    )
    parser.add_argument(
        "--adapter",
        help="Extract LoRA adapter from PEFT directory → <adapter_dir>-TRNQ/",
    )
    args = parser.parse_args()

    if not args.model and not args.adapter:
        parser.print_help()
        raise ValueError("specify --model and/or --adapter <adapter_dir>")

    model_dir = args.model_dir
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found.")

    binary_path = _find_quantize_binary()

    print("=" * 60)
    print("Trillim Quantizer")
    print("=" * 60)

    # Parse model config
    print(f"\nParsing {config_path}...")
    config = ModelConfig.from_config_json(config_path, model_dir)

    print(f"  Architecture: {config.arch_type.name}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Intermediate dim: {config.intermediate_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")

    # Set up output directories
    adapter_output_dir = None
    model_output_dir = None

    if args.model:
        model_output_dir = _make_model_output_dir(model_dir)
        print(f"\n  Model output: {model_output_dir}")

    if args.adapter:
        adapter_dir = args.adapter
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        _validate_adapter_dims(adapter_dir, config)
        adapter_output_dir = _make_adapter_output_dir(adapter_dir)
        print(f"  Adapter output: {adapter_output_dir}")

    if args.model:
        # Quantize model weights (+ LoRA if adapter provided)
        print("\n" + "-" * 60)
        print("Quantizing model weights...")
        print("-" * 60)

        _run_cpp_quantizer(
            binary_path, model_dir, config, model_output_dir,
            adapter_dir=args.adapter if args.adapter else None,
            adapter_output_dir=adapter_output_dir,
        )

        qmodel_path = os.path.join(model_output_dir, "qmodel.tensors")
        rope_path = os.path.join(model_output_dir, "rope.cache")
        print(f"\n  Written: {qmodel_path}")
        print(f"  Written: {rope_path}")

        if args.adapter:
            lora_path = os.path.join(adapter_output_dir, "qmodel.lora")
            print(f"  Written: {lora_path}")

        # Copy model files and write config to TRNQ dir
        _copy_model_files(model_dir, model_output_dir)
        _write_trillim_model_config(model_output_dir, config, model_dir)

    elif args.adapter:
        # LoRA-only extraction
        print("\n" + "-" * 60)
        print("Extracting LoRA adapter...")
        print("-" * 60)

        _run_cpp_lora_only(
            binary_path, model_dir, config, args.adapter,
            adapter_output_dir=adapter_output_dir,
        )

        lora_path = os.path.join(adapter_output_dir, "qmodel.lora")
        print(f"\n  Written: {lora_path}")

    # Copy tokenizer files and write trillim_config.json to adapter output dir
    if args.adapter:
        _copy_adapter_tokenizer_files(args.adapter, adapter_output_dir)
        _write_trillim_adapter_config(adapter_output_dir, config, args.adapter, model_dir)

    print("\n" + "=" * 60)
    print("Done!")
    if model_output_dir:
        print(f"\nQuantized model ready at: {model_output_dir}")
        if adapter_output_dir:
            print(f"Usage: trillim chat {model_output_dir} --lora {adapter_output_dir}")
        else:
            print(f"Usage: trillim chat {model_output_dir}")
    elif adapter_output_dir:
        print(f"\nQuantized adapter ready at: {adapter_output_dir}")
        print(f"Usage: trillim chat {model_dir} --lora {adapter_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
