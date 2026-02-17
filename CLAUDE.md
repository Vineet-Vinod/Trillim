# CLAUDE.md

## Project Overview

Trillim Python SDK — MIT-licensed package for running BitNet-architecture models on CPUs. Communicates with the proprietary DarkNet C++ inference engine via a subprocess stdin/stdout protocol.

The C++ engine lives in a sibling repo (`../DarkNet`). Compiled binaries are bundled into `src/trillim/_bin/` via `make bundle-binaries`.

## Gotchas

- Use `uv` for all Python commands. The Makefile lives in `../DarkNet`; run `make help` there for available targets.
- **Under 12GB system memory.** Be careful when loading safetensors files to avoid OOM.
- The stdin protocol between Python and C++ must stay in sync. See `PROTOCOL.md` for the full spec.
  - If the C++ side changes sampling parameters, update both `src/trillim/inference.py` and `src/trillim/server/_llm.py` to match.
  - Same for `--config` CLI args: `_config_args()` in `src/trillim/inference.py` must match `parse_config()` in DarkNet's `src/inference.cpp`.
- KV cache matching uses **string-level** comparison of rendered chat templates, NOT token-level. This is intentional — re-tokenizing a full conversation produces different tokens than incremental tokenization. Do not change to token-level for chat completions.
- Binary paths are resolved via `src/trillim/_bin_path.py`. Never hardcode paths to executables.
- **Never publish an sdist** (`uv build` builds sdist + wheel). Always use `uv build --wheel` or `scripts/build_wheels.py`. An sdist can't produce working binaries — the C++ engine is proprietary and not in this repo.
- Voices default to `~/.trillim/voices/`. The `--voices-dir` CLI flag overrides this.
- For LoRA: if output doesn't follow the fine-tuning dataset, check `chat_template.jinja`. A buggy template produces garbage.
- The API server does NOT auto-reload. Restart after code changes.
- When using `timeout` for debugging inference, keep it <= 30 seconds.
