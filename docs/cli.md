# CLI Reference

The `trillim` CLI is the quickest way to pull bundles, quantize local checkpoints, open an interactive chat shell, or run the demo API server.

If you installed Trillim with `uv`, prefix the commands on this page with `uv run`.

## Core Rules

- `trillim pull` only downloads from the `Trillim/<name>` namespace on Hugging Face.
- `trillim chat` and `trillim serve` load by store ID only:
  - `Trillim/<name>`
  - `Local/<name>`
- `trillim quantize` is the command that accepts raw local filesystem paths.
- `--trust-remote-code` is opt-in. Bundles that declare remote code fail closed unless you pass it.

## Command Summary

| Command | What it does |
| --- | --- |
| `trillim models` | List published bundles in the `Trillim` org |
| `trillim pull` | Download one published bundle into `~/.trillim/models/Trillim/` |
| `trillim list` | List the bundles you already have locally |
| `trillim chat` | Start a local multi-turn chat shell |
| `trillim serve` | Start the demo HTTP server on `127.0.0.1:8000` |
| `trillim quantize` | Quantize a local model or local adapter into `~/.trillim/models/Local/` |

## `trillim models`

List published bundles from the `Trillim` Hugging Face organization.

```bash
trillim models
```

Use this when you want to see what can be pulled. It requires network access.

The `STATUS` column shows:

- `local` for downloaded bundles under `~/.trillim/models/Trillim/` whose `format_version` exactly matches the current Trillim version
- `stale` for downloaded bundles under `~/.trillim/models/Trillim/` whose `format_version` no longer matches

## `trillim pull`

Download one published bundle.

```bash
trillim pull Trillim/BitNet-TRNQ
trillim pull Trillim/BitNet-TRNQ --revision main
trillim pull Trillim/BitNet-TRNQ --force
```

Flags:

| Flag | Meaning |
| --- | --- |
| `model_id` | Required Hugging Face ID in the form `Trillim/<name>` |
| `--revision` | Branch, tag, or commit hash |
| `--force`, `-f` | Re-download even if the bundle already exists locally |

`pull` does not support arbitrary Hugging Face repos. If you need your own checkpoint, quantize it locally and load the resulting `Local/...` bundle.

## `trillim list`

List what Trillim can already load on this machine.

```bash
trillim list
```

The output is split into:

- `Downloaded`: bundles under `~/.trillim/models/Trillim/`
- `Local`: bundles under `~/.trillim/models/Local/`

## `trillim chat`

Open a simple multi-turn chat shell.

```bash
trillim chat Trillim/BitNet-TRNQ
trillim chat Trillim/BitNet-TRNQ Trillim/BitNet-GenZ-LoRA-TRNQ
trillim chat Trillim/BitNet-Large-TRNQ --trust-remote-code
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `model_dir` | Required store ID for the base model |
| `adapter_dir` | Optional store ID for a LoRA adapter |

Flags:

| Flag | Meaning |
| --- | --- |
| `--trust-remote-code` | Allow bundles that reference custom tokenizer or config code |

Interactive controls:

| Input | Meaning |
| --- | --- |
| `/new` | Start a new conversation |
| `q` | Quit |
| `Ctrl+G` | Open the current prompt in your editor |

Notes:

- The chat shell keeps conversation history until you reset it.
- It does not expose sampling, search, or hot-swap flags in this implementation.
- If you need search, custom sampling, or HTTP access, use the Python SDK or the server surface.

## `trillim serve`

Start the demo HTTP server.

```bash
trillim serve Trillim/BitNet-TRNQ
trillim serve Trillim/BitNet-TRNQ --voice
trillim serve Trillim/BitNet-Large-TRNQ --trust-remote-code
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `model_dir` | Required store ID for the base model |

Flags:

| Flag | Meaning |
| --- | --- |
| `--voice` | Add `STT` and `TTS` routes alongside the LLM routes |
| `--trust-remote-code` | Allow bundles that reference custom tokenizer or config code |

Important constraints:

- The CLI server binds to `127.0.0.1:8000`.
- The CLI server does not expose host/port flags.
- The CLI server does not enable `/v1/models/swap`.
- Search is not configured from the CLI in this repo.

If you need custom host/port, hot swap, or search configuration, build `Server(...)` in Python instead.

## `trillim quantize`

Quantize a local checkpoint into Trillim's managed `Local/` store.

```bash
# Quantize a base model
trillim quantize /path/to/model

# Quantize a LoRA adapter against a base model
trillim quantize /path/to/base-model /path/to/adapter
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `model_dir` | Required local filesystem path to the base model directory |
| `adapter_dir` | Optional local filesystem path to a LoRA adapter directory |

Output:

- Model bundles are written under `~/.trillim/models/Local/<model-name>-TRNQ`
- Adapter bundles are written under `~/.trillim/models/Local/<adapter-name>-TRNQ`

If the second positional argument is present, the command quantizes the adapter. It does not quantize both outputs in one invocation. Run the command twice if you want both the model bundle and the adapter bundle.

Additional notes:

- Bonsai source checkpoints should include `trillim_source.json` in the model root with `"architecture": "bonsai"` or `"architecture": "bonsai_ternary"`.
- Legacy README-based Bonsai detection still works for older checkpoints, but it is deprecated.
- Qwen3-based Bonsai checkpoints are supported, including Bonsai 1-bit (binary) and grouped-ternary bundles.
- Bonsai bundles use binary or grouped-ternary quantization metadata. Existing BitNet-style flows continue to use ternary quantization metadata.
- The managed store naming does not change for Bonsai bundles: the output still lands under `Local/...-TRNQ`.
- Bonsai bundles are loaded with the same store IDs and commands as any other Trillim bundle, including `chat`, `serve`, and the Python SDK.
- The source directories must be outside `~/.trillim/models/`.
- If the target bundle name already exists, Trillim either asks before overwriting or writes a deduped name such as `...-TRNQ-2`.
- Known Qwen 3.5 multimodal checkpoints are quantized in text-only mode.

## Most Common CLI Confusions

- `models` is the remote listing command. `list` is the local listing command.
- `chat` and `serve` load store IDs, not raw paths.
- `quantize` takes raw paths, not store IDs.
- The implemented server has `/v1/chat/completions`, not `/v1/completions`.
