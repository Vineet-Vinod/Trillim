"""Managed inference subprocess protocol for LLM generation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from trillim.components.llm._config import InitConfig, ModelRuntimeConfig, SamplingDefaults


class EngineError(RuntimeError):
    """Base class for inference worker failures."""


class EngineCrashedError(EngineError):
    """Raised when the inference worker dies unexpectedly."""


class EngineProtocolError(EngineError):
    """Raised when the inference worker violates the wire protocol."""


class EngineProgressTimeoutError(EngineError, TimeoutError):
    """Raised when the inference worker stops making progress."""


class InferenceEngine:
    """Drive the inference subprocess for one active model."""

    def __init__(
        self,
        model: ModelRuntimeConfig,
        tokenizer,
        defaults: SamplingDefaults,
        *,
        init_config: InitConfig | None = None,
        progress_timeout: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.defaults = defaults
        self.init_config = (
            InitConfig(model_dir=model.path)
            if init_config is None
            else init_config
        )
        self.progress_timeout = progress_timeout
        self.binary_path = _bundled_binary_path()
        self.process: asyncio.subprocess.Process | None = None
        self._cached_token_ids: tuple[int, ...] = ()

    async def start(self) -> None:
        """Start the inference worker and send its init block."""
        if self.process is not None and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            self.binary_path,
            str(self.model.path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._write_block(_build_init_block(self.model, self.init_config))
        except Exception:
            await self._kill_process()
            raise

    async def stop(self) -> None:
        """Stop the worker and clear cached prompt state."""
        self._cached_token_ids = ()
        process = self.process
        self.process = None
        if process is None or process.returncode is not None:
            return
        try:
            assert process.stdin is not None
            process.stdin.write(b"0\n")
            await asyncio.wait_for(process.stdin.drain(), timeout=self.progress_timeout)
            await asyncio.wait_for(process.wait(), timeout=self.progress_timeout)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError, OSError):
            try:
                process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    async def recover(self) -> None:
        """Restart the worker after a failed or cancelled generation."""
        await self.stop()
        await self.start()

    async def generate(
        self,
        token_ids: Sequence[int],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[int]:
        """Generate tokens from the worker and update prompt-cache state."""
        process = self._require_running()
        request_tokens = tuple(token_ids)
        kv_position = _common_prefix_len(self._cached_token_ids, request_tokens)
        completed = False
        generated: list[int] = []
        try:
            await self._write_block(
                _build_request_block(
                    kv_position=kv_position,
                    tokens=request_tokens[kv_position:],
                    temperature=self.defaults.temperature if temperature is None else temperature,
                    top_k=self.defaults.top_k if top_k is None else top_k,
                    top_p=self.defaults.top_p if top_p is None else top_p,
                    repetition_penalty=(
                        self.defaults.repetition_penalty
                        if repetition_penalty is None
                        else repetition_penalty
                    ),
                    rep_penalty_lookback=(
                        self.defaults.rep_penalty_lookback
                        if rep_penalty_lookback is None
                        else rep_penalty_lookback
                    ),
                    max_tokens=self.defaults.max_tokens if max_tokens is None else max_tokens,
                )
            )
            while True:
                raw = await self._readline("token_id")
                token_id = _parse_protocol_int(raw, "token_id")
                generated.append(token_id)
                if token_id in self.model.eos_tokens:
                    break
                yield token_id
            kv_line = await self._readline("kv_position")
            kv_position = _parse_protocol_int(kv_line, "kv_position")
            combined = request_tokens + tuple(generated)
            if kv_position < 0 or kv_position > len(combined):
                raise EngineProtocolError(
                    f"Invalid kv_position {kv_position}; expected a value between 0 and {len(combined)}"
                )
            self._cached_token_ids = combined[:kv_position]
            completed = True
        except (asyncio.CancelledError, GeneratorExit):
            self._cached_token_ids = ()
            await self._kill_process()
            raise
        except EngineError:
            self._cached_token_ids = ()
            await self._kill_process()
            raise
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self._cached_token_ids = ()
            await self._kill_process()
            raise EngineCrashedError("Inference engine crashed") from exc
        finally:
            if not completed:
                self._cached_token_ids = ()
                if process.returncode is not None:
                    self.process = None

    def _require_running(self) -> asyncio.subprocess.Process:
        process = self.process
        if process is None or process.returncode is not None:
            raise EngineCrashedError("Inference process is not running")
        return process

    async def _write_block(self, block: str) -> None:
        process = self._require_running()
        stdin = process.stdin
        if stdin is None:
            raise EngineCrashedError("Inference process stdin is unavailable")
        stdin.write(block.encode("utf-8"))
        try:
            await asyncio.wait_for(stdin.drain(), timeout=self.progress_timeout)
        except asyncio.TimeoutError as exc:
            raise EngineProgressTimeoutError(
                f"Inference engine made no write progress for {self.progress_timeout} seconds"
            ) from exc

    async def _readline(self, field_name: str) -> bytes:
        process = self._require_running()
        stdout = process.stdout
        if stdout is None:
            raise EngineCrashedError("Inference process stdout is unavailable")
        try:
            line = await asyncio.wait_for(
                stdout.readline(),
                timeout=self.progress_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise EngineProgressTimeoutError(
                f"Inference engine made no {field_name} progress for {self.progress_timeout} seconds"
            ) from exc
        if not line:
            stderr = await _read_stderr(process)
            if stderr:
                raise EngineCrashedError(f"Inference engine crashed: {stderr}")
            raise EngineCrashedError("Inference engine crashed")
        return line

    async def _kill_process(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is not None:
            self.process = None
            return
        try:
            process.kill()
        except OSError:
            pass
        await process.wait()
        self.process = None


async def _read_stderr(process: asyncio.subprocess.Process) -> str:
    stderr = process.stderr
    if stderr is None:
        return ""
    try:
        return (await stderr.read()).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _bundled_binary_path() -> str:
    suffix = ".exe" if os.name =="nt" else ""
    bin_dir = Path(__file__).resolve().parents[2] / "_bin"
    bundled = bin_dir / f"trillim-inference{suffix}"
    if bundled.is_file():
        return str(bundled)
    if suffix:
        fallback = bin_dir / "trillim-inference"
        if fallback.is_file():
            return str(fallback)
    raise FileNotFoundError(f"Missing bundled LLM inference binary: {bundled}")


def _build_init_block(model: ModelRuntimeConfig, init_config: InitConfig) -> str:
    pairs = [
        f"arch_type={int(model.arch_type)}",
        f"activation={int(model.activation)}",
        f"hidden_dim={model.hidden_dim}",
        f"intermediate_dim={model.intermediate_dim}",
        f"num_layers={model.num_layers}",
        f"num_heads={model.num_heads}",
        f"num_kv_heads={model.num_kv_heads}",
        f"vocab_size={model.vocab_size}",
        f"head_dim={model.head_dim}",
        f"max_position_embeddings={model.max_position_embeddings}",
        f"norm_eps={model.norm_eps}",
        f"rope_theta={model.rope_theta}",
        f"tie_word_embeddings={1 if model.tie_word_embeddings else 0}",
        f"has_qkv_bias={1 if model.has_qkv_bias else 0}",
        f"has_attn_sub_norm={1 if model.has_attn_sub_norm else 0}",
        f"has_ffn_sub_norm={1 if model.has_ffn_sub_norm else 0}",
        f"eos_tokens={','.join(str(token_id) for token_id in model.eos_tokens)}",
    ]
    if init_config.num_threads:
        pairs.append(f"num_threads={init_config.num_threads}")
    if init_config.lora_dir is not None:
        pairs.append(f"lora_dir={_first_protocol_line(str(init_config.lora_dir))}")
    if init_config.lora_quant is not None:
        pairs.append(f"lora_quant={_first_protocol_line(init_config.lora_quant)}")
    if init_config.unembed_quant is not None:
        pairs.append(f"unembed_quant={_first_protocol_line(init_config.unembed_quant)}")
    return f"{len(pairs)}\n" + "".join(f"{pair}\n" for pair in pairs)


def _first_protocol_line(value: str) -> str:
    lines = value.strip().splitlines()
    if not lines:
        return value
    return lines[0]


def _build_request_block(
    *,
    kv_position: int,
    tokens: Sequence[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    rep_penalty_lookback: int,
    max_tokens: int,
) -> str:
    pairs = [
        f"kv_position={kv_position}",
        f"tokens={','.join(str(token_id) for token_id in tokens)}",
        f"temperature={temperature}",
        f"top_k={top_k}",
        f"top_p={top_p}",
        f"repetition_penalty={repetition_penalty}",
        f"rep_penalty_lookback={rep_penalty_lookback}",
        f"max_tokens={max_tokens}",
    ]
    return f"{len(pairs)}\n" + "".join(f"{pair}\n" for pair in pairs)


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    shared = 0
    limit = min(len(left), len(right))
    while shared < limit and left[shared] == right[shared]:
        shared += 1
    return shared


def _parse_protocol_int(raw: bytes, field_name: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise EngineProtocolError(
            f"Protocol error: expected int {field_name}, got {raw.strip()!r}"
        ) from exc
