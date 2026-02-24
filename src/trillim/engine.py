# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Async InferenceEngine — shared by CLI and API server."""

import asyncio
import os
from collections.abc import AsyncGenerator

_ENGINE_TIMEOUT = 300  # seconds; maximum wait for a single engine I/O operation


class InferenceEngine:
    """Manages the C++ inference subprocess with async I/O."""

    def __init__(
        self,
        model_dir: str,
        tokenizer,
        stop_tokens: set[int],
        default_params: dict,
        arch_config,
        adapter_dir: str | None = None,
        num_threads: int = 0,
        lora_quant: str | None = None,
        unembed_quant: str | None = None,
    ):
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens
        self.default_params = default_params
        self.arch_config = arch_config
        self.adapter_dir = adapter_dir
        self.num_threads = num_threads
        self.lora_quant = lora_quant
        self.unembed_quant = unembed_quant
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.cached_token_ids: list[int] = []
        self._last_cache_hit: int = 0
        self._cached_prompt_str: str = ""

    async def start(self):
        """Launch the C++ inference subprocess."""
        from trillim.utils import _build_init_config, load_engine_options

        if self.adapter_dir:
            trillim_cfg_path = os.path.join(self.adapter_dir, "trillim_config.json")
            if not os.path.exists(trillim_cfg_path):
                raise RuntimeError(
                    f"{trillim_cfg_path} not found. "
                    "This adapter has not been quantized for Trillim. "
                    f"Run: trillim quantize <model_dir> --adapter {self.adapter_dir}"
                )
            lora_path = os.path.join(self.adapter_dir, "qmodel.lora")
            if not os.path.exists(lora_path):
                raise RuntimeError(
                    f"--lora set but {lora_path} not found. "
                    f"Run: trillim quantize <model_dir> --adapter {self.adapter_dir}"
                )
            from trillim.model_store import validate_adapter_model_compat
            validate_adapter_model_compat(self.adapter_dir, self.model_dir)

        from trillim._bin_path import inference_bin

        cmd = [inference_bin(), self.model_dir]
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        engine_options = load_engine_options(
            num_threads=self.num_threads,
            lora_quant=self.lora_quant,
            unembed_quant=self.unembed_quant,
        )
        init_block = _build_init_config(
            self.arch_config, adapter_dir=self.adapter_dir, **engine_options,
        )
        self.process.stdin.write(init_block.encode())
        await self.process.stdin.drain()

    async def stop(self):
        """Shut down the subprocess gracefully."""
        self.cached_token_ids = []
        self._last_cache_hit = 0
        self._cached_prompt_str = ""
        if self.process and self.process.returncode is None:
            try:
                self.process.stdin.write(b"0\n")
                await self.process.stdin.drain()
                await asyncio.wait_for(self.process.wait(), timeout=300)
            except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError, OSError):
                self.process.kill()
                await self.process.wait()

    async def generate(
        self,
        token_ids: list[int],
        prompt_str: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[int, None]:
        """Send tokens + params to the C++ process and yield generated token IDs."""
        d = self.default_params
        temp = temperature if temperature is not None else d["temperature"]
        tk = top_k if top_k is not None else d["top_k"]
        tp = top_p if top_p is not None else d["top_p"]
        rp = (
            repetition_penalty
            if repetition_penalty is not None
            else d["repetition_penalty"]
        )
        rl = (
            rep_penalty_lookback
            if rep_penalty_lookback is not None
            else d["rep_penalty_lookback"]
        )
        mt = max_tokens if max_tokens is not None else 0

        async with self.lock:
            proc = self.process
            if proc is None or proc.returncode is not None:
                raise RuntimeError("Inference process is not running")

            # String-level prefix matching (mirrors CLI chat approach):
            # Compare rendered prompt strings to avoid re-tokenization mismatches.
            if (
                prompt_str is not None
                and self._cached_prompt_str
                and prompt_str.startswith(self._cached_prompt_str)
            ):
                suffix = prompt_str[len(self._cached_prompt_str) :]
                delta_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
                all_token_ids = self.cached_token_ids + delta_tokens
                reset_flag = 0
                match_len = len(self.cached_token_ids)
            elif prompt_str is not None:
                # String mismatch — different conversation, full reset.
                # Do NOT fall through to token-level matching: shared
                # template-header tokens could trick it into reusing a
                # stale KV cache from the previous conversation.
                delta_tokens = token_ids
                reset_flag = 1
                match_len = 0
                all_token_ids = token_ids
            else:
                # Token-level prefix matching (for /v1/completions with no template)
                cached = self.cached_token_ids
                match_len = 0
                limit = min(len(token_ids), len(cached))
                while match_len < limit and token_ids[match_len] == cached[match_len]:
                    match_len += 1

                if match_len > 0 and match_len == len(cached):
                    delta_tokens = token_ids[match_len:]
                    reset_flag = 0
                else:
                    delta_tokens = token_ids
                    reset_flag = 1
                    match_len = 0
                all_token_ids = token_ids

            self._last_cache_hit = match_len

            # Build count-prefixed key=value request block
            from trillim.utils import _build_request_block

            req_block = _build_request_block(
                delta_tokens, reset_flag,
                temperature=temp, top_k=tk, top_p=tp,
                repetition_penalty=rp, rep_penalty_lookback=rl,
                max_tokens=mt or None,
            )
            try:
                proc.stdin.write(req_block.encode())
                await asyncio.wait_for(proc.stdin.drain(), timeout=_ENGINE_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                raise RuntimeError(
                    f"Inference engine unresponsive for {_ENGINE_TIMEOUT}s — "
                    "the model may be too large for available memory"
                )
            except (BrokenPipeError, ConnectionResetError, OSError):
                stderr = ""
                try:
                    stderr_bytes = await proc.stderr.read()
                    stderr = stderr_bytes.decode().strip()
                except Exception:
                    pass
                msg = "Inference engine crashed"
                if stderr:
                    msg += f": {stderr}"
                raise RuntimeError(msg)

            # Read generated tokens until EOS
            generated_tokens: list[int] = []
            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=_ENGINE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    raise RuntimeError(
                        f"Inference engine unresponsive for {_ENGINE_TIMEOUT}s — "
                        "the model may be too large for available memory"
                    )
                if not line:
                    break
                try:
                    token_id = int(line.strip())
                except ValueError:
                    proc.kill()
                    raise RuntimeError(
                        f"Protocol error: expected int token_id, got {line.strip()!r}"
                    )
                generated_tokens.append(token_id)
                if token_id in self.stop_tokens:
                    break
                yield token_id

            # Read kv_position and update cache state
            try:
                kv_line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=_ENGINE_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                self.cached_token_ids = []
                self._cached_prompt_str = ""
                raise RuntimeError(
                    f"Inference engine unresponsive for {_ENGINE_TIMEOUT}s — "
                    "the model may be too large for available memory"
                )
            if kv_line:
                try:
                    kv_position = int(kv_line.strip())
                except ValueError:
                    proc.kill()
                    raise RuntimeError(
                        f"Protocol error: expected int kv_position, got {kv_line.strip()!r}"
                    )
                self.cached_token_ids = (all_token_ids + generated_tokens)[:kv_position]
            else:
                self.cached_token_ids = []
                self._cached_prompt_str = ""
