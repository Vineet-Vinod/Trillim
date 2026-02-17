# PROTOCOL.md — DarkNet ↔ Trillim Interface Specification

**Protocol version:** 1

This document defines the interface contract between the DarkNet C++ inference
engine and the Trillim Python SDK. Both repos must stay in sync. Bump the
protocol version when either side changes the wire format.

---

## 1. Binary Invocation

```
inference <model_dir> [--lora] --config <args...>
```

The Python side launches the C++ binary as a subprocess with stdin/stdout pipes.

---

## 2. `--config` Argument Order

Positional arguments after `--config`, in this exact order:

| # | Field | Type | Python source |
|---|-------|------|---------------|
| 1 | `arch_type` | int | `int(arch_config.arch_type)` |
| 2 | `activation` | int | `int(arch_config.arch_info.activation)` |
| 3 | `hidden_dim` | int | `arch_config.hidden_dim` |
| 4 | `intermediate_dim` | int | `arch_config.intermediate_dim` |
| 5 | `num_layers` | int | `arch_config.num_layers` |
| 6 | `num_heads` | int | `arch_config.num_heads` |
| 7 | `num_kv_heads` | int | `arch_config.num_kv_heads` |
| 8 | `vocab_size` | int | `arch_config.vocab_size` |
| 9 | `head_dim` | int | `arch_config.head_dim` |
| 10 | `max_position_embeddings` | int | `arch_config.max_position_embeddings` |
| 11 | `norm_eps` | float | `arch_config.norm_eps` |
| 12 | `rope_theta` | float | `arch_config.rope_theta` |
| 13 | `tie_word_embeddings` | 0/1 | `arch_config.tie_word_embeddings` |
| 14 | `has_qkv_bias` | 0/1 | `arch_config.has_qkv_bias` |
| 15 | `has_attn_sub_norm` | 0/1 | `arch_config.arch_info.has_attn_sub_norm` |
| 16 | `has_ffn_sub_norm` | 0/1 | `arch_config.arch_info.has_ffn_sub_norm` |
| 17 | `num_threads` | int | 0 = auto (`hardware_concurrency - 2`) |
| 18 | `num_eos_tokens` | int | `len(arch_config.eos_tokens)` |
| 19+ | `eos_token_id` ... | int | one per EOS token (max 8) |

**Python:** `_config_args()` in `inference.py`
**C++:** `parse_config()` in `src/inference.cpp`

---

## 3. Stdin/Stdout Protocol (per request)

### Python → C++ (stdin)

Each value is a newline-terminated string. Order matters.

```
<num_tokens>          # int — number of new token IDs to send (0 = quit)
<reset_flag>          # int — 1 = reset KV cache, 0 = continue
<temperature>         # float
<top_k>               # int
<top_p>               # float
<repetition_penalty>  # float
<rep_penalty_lookback> # int
<max_tokens>          # int — 0 = unlimited
<token_id_1>          # int — repeated num_tokens times
<token_id_2>
...
```

To terminate the process, send `0\n` as `num_tokens`.

### C++ → Python (stdout)

```
<generated_token_1>   # int — one per line
<generated_token_2>
...                   # continues until EOS token or max_tokens reached
<kv_position>         # int — total tokens in KV cache after generation
```

The EOS token is included in the output stream. The `kv_position` line always
follows the last generated token (including EOS).

---

## 4. Enum Tables

### ArchType

| Value | Name | Description |
|-------|------|-------------|
| 0 | UNKNOWN | — |
| 1 | BITNET | BitNet with ternary weights, ReLU² |
| 2 | LLAMA | Llama-style, SiLU activation |

### ActivationType

| Value | Name | Description |
|-------|------|-------------|
| 0 | RELU_SQR | ReLU squared (BitNet) |
| 1 | SILU | SiLU / Swish (Llama, Qwen, Mistral) |

**Python:** `ArchType`, `ActivationType` in `model_arch.py`
**C++:** `ArchType`, `ActivationType` in `include/model/config.h`

---

## 5. File Format Headers

### qmodel.tensors (TRLM)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `TRLM` (0x54524C4D) |
| 4 | 4 | Version: `1` (uint32 LE) |
| 8 | 4 | Architecture: ArchType (uint32 LE) |
| 12 | 4 | Reserved: `0` (uint32 LE) |

### rope.cache (TRRC)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `TRRC` (0x54525243) |
| 4 | 4 | Version: `1` (uint32 LE) |
| 8 | 4 | Reserved: `0` (uint32 LE) |

### qmodel.lora (TRLA)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `TRLA` (0x54524C41) |
| 4 | 4 | Version: `1` (uint32 LE) |
| 8 | 4 | Reserved: `0` (uint32 LE) |

**Python:** `MAGIC_TENSORS`, `MAGIC_ROPE`, `MAGIC_LORA` in `model_arch.py`
**C++:** `validate_tensors_header()`, `validate_rope_header()`, `validate_lora_header()` in `include/trillim/file_header.h`

---

## 6. Quantize Manifest Binary Layout

Written by Python (`quantize.py:write_manifest()`), read by C++ (`trillim-quantize`).

### Header

| Size | Type | Field |
|------|------|-------|
| 2 | uint16 LE | Number of shard files |

Per shard file:
| 2 | uint16 LE | Path length in bytes |
| N | UTF-8 bytes | Shard file path |

### Tensor Entries

| Size | Type | Field |
|------|------|-------|
| 4 | uint32 LE | Number of tensor entries |

Per tensor entry (59 bytes each):
| Size | Type | Field |
|------|------|-------|
| 1 | uint8 | Action code |
| 1 | uint8 | DType code |
| 4 | uint32 LE | Rows |
| 4 | uint32 LE | Cols |
| 4 | uint32 LE | Padded rows |
| 4 | uint32 LE | Padded cols |
| 2 | uint16 LE | Shard index |
| 8 | uint64 LE | Data offset |
| 8 | uint64 LE | Data size |
| 1 | uint8 | Has scale (0/1) |
| 2 | uint16 LE | Scale shard index |
| 8 | uint64 LE | Scale offset |
| 8 | uint64 LE | Scale size |

### Action Codes

| Value | Name | Description |
|-------|------|-------------|
| 0 | BF16_RAW | Copy bf16 tensor as-is |
| 1 | TERNARY_QUANTIZE | Quantize bf16 → ternary |
| 2 | EMBEDDING_I8 | Quantize embedding to int8 |
| 3 | REPACK_TERNARY | Repack pre-quantized i8/u8 ternary |

### DType Codes

| Value | Name | Element size |
|-------|------|-------------|
| 0 | F32 | 4 bytes |
| 1 | F16 | 2 bytes |
| 2 | BF16 | 2 bytes |
| 3 | I8 | 1 byte |
| 4 | U8 | 1 byte |

### LoRA Entries (optional, appended after tensor entries)

| Size | Type | Field |
|------|------|-------|
| 4 | uint32 LE | Number of layers |
| 4 | uint32 LE | Targets per layer |
| 8 | float64 LE | LoRA scale (alpha / rank) |

Per layer × per target:
| 1 | uint8 | Present (0 = absent, 1 = present) |

If present:
| Size | Type | Field |
|------|------|-------|
| 1 | uint8 | A dtype code |
| 4 | uint32 LE | A rows |
| 4 | uint32 LE | A cols |
| 2 | uint16 LE | A shard index |
| 8 | uint64 LE | A offset |
| 8 | uint64 LE | A size |
| 1 | uint8 | B dtype code |
| 4 | uint32 LE | B rows |
| 4 | uint32 LE | B cols |
| 2 | uint16 LE | B shard index |
| 8 | uint64 LE | B offset |
| 8 | uint64 LE | B size |

---

## 7. LoRA Target Order

Per-layer LoRA targets in `qmodel.lora`, in this exact order:

| Index | Target |
|-------|--------|
| 0 | `self_attn.k_proj` |
| 1 | `self_attn.v_proj` |
| 2 | `self_attn.q_proj` |
| 3 | `self_attn.o_proj` |
| 4 | `mlp.gate_proj` |
| 5 | `mlp.up_proj` |
| 6 | `mlp.down_proj` |

**Python:** `LORA_TARGETS` in `model_arch.py`
