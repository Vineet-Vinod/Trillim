# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Async InferenceEngine — shared by CLI and API server."""

import asyncio
import os
from collections.abc import AsyncGenerator

from pydantic import ValidationError

from trillim._prompt_cache import PromptCacheManager, PromptSnapshot
from trillim._sampling import first_validation_error

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
        self._prompt_cache = PromptCacheManager()

    @property
    def cached_token_ids(self) -> list[int]:
        return list(self._prompt_cache.token_ids)

    @property
    def cached_prompt_str(self) -> str | None:
        return self._prompt_cache.prompt_str

    @property
    def last_cache_hit(self) -> int:
        return self._prompt_cache.last_cache_hit

    def reset_prompt_cache(self) -> None:
        self._prompt_cache.clear()

    def finalize_prompt_cache(self, snapshot: PromptSnapshot) -> None:
        self._prompt_cache.finalize_prompt(snapshot)

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
        self.reset_prompt_cache()
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
        request = PromptSnapshot.create(token_ids, prompt_str)

        async with self.lock:
            proc = self.process
            if proc is None or proc.returncode is not None:
                raise RuntimeError("Inference process is not running")
            completed = False

            async def _raise_engine_crash() -> None:
                self.reset_prompt_cache()
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

            try:
                plan = self._prompt_cache.plan(
                    request,
                    encode_suffix=lambda suffix: self.tokenizer.encode(
                        suffix,
                        add_special_tokens=False,
                    ),
                )

                # Build count-prefixed key=value request block
                from trillim.utils import _build_request_block

                try:
                    req_block = _build_request_block(
                        list(plan.delta_tokens),
                        plan.reset_flag,
                        temperature=temp,
                        top_k=tk,
                        top_p=tp,
                        repetition_penalty=rp,
                        rep_penalty_lookback=rl,
                        max_tokens=mt,
                    )
                except ValidationError as exc:
                    raise ValueError(first_validation_error(exc)) from exc
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
                    await _raise_engine_crash()

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
                        await _raise_engine_crash()
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
                    self.reset_prompt_cache()
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
                            "Protocol error: expected int kv_position, "
                            f"got {kv_line.strip()!r}"
                        )
                    self._prompt_cache.commit_generation(
                        plan,
                        generated_token_ids=generated_tokens,
                        kv_position=kv_position,
                    )
                else:
                    await _raise_engine_crash()
                completed = True
            finally:
                if not completed:
                    self.reset_prompt_cache()
                    if proc.returncode is None:
                        try:
                            proc.kill()
                        except OSError:
                            pass
