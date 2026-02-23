# CLAUDE.md

## Project Overview

Trillim Python SDK — MIT-licensed package for running BitNet-architecture models on CPUs. Communicates with the proprietary DarkNet C++ inference engine via a subprocess stdin/stdout protocol.

The C++ engine lives in a sibling repo (`../DarkNet`). Compiled binaries are bundled into `src/trillim/_bin/` via `make bundle-binaries`.

## Gotchas

- Use `uv` for all Python commands. The Makefile lives in `../DarkNet`; run `make help` there for available targets.
- **Under 12GB system memory.** Be careful when loading safetensors files to avoid OOM.
- The stdin protocol between Python and C++ uses count-prefixed `key=value\n` blocks. Both the init config (sent once at startup) and per-request blocks use this format. Both sides ignore unknown keys (forward-compatible), so they don't need to stay in sync.
  - Init config is built by `_build_init_config()` and request blocks by `_build_request_block()` in `src/trillim/inference.py`.
- KV cache matching uses **string-level** comparison of rendered chat templates, NOT token-level. This is intentional — re-tokenizing a full conversation produces different tokens than incremental tokenization. Do not change to token-level for chat completions.
- Binary paths are resolved via `src/trillim/_bin_path.py`. Never hardcode paths to executables.
- **Never publish an sdist** (`uv build` builds sdist + wheel). Always use `uv build --wheel` or `scripts/build_wheels.py`. An sdist can't produce working binaries — the C++ engine is proprietary and not in this repo.
- Voices default to `~/.trillim/voices/`. The `--voices-dir` CLI flag overrides this.
- For LoRA: if output doesn't follow the fine-tuning dataset, check `chat_template.jinja`. A buggy template produces garbage.
- The API server does NOT auto-reload. Restart after code changes.
- When using `timeout` for debugging inference, keep it <= 30 seconds.
