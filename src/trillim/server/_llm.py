# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""LLM component — wraps InferenceEngine and exposes inference routes."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ._component import Component
from ._helpers import load_default_params, make_id, now
from ._models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    ModelListResponse,
    ServerState,
    UsageInfo,
)


# ---------------------------------------------------------------------------
# InferenceEngine (unchanged from server.py)
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Manages the C++ inference subprocess with async I/O."""

    def __init__(
        self,
        model_dir: str,
        tokenizer,
        stop_tokens: set[int],
        default_params: dict,
        arch_config,
        use_lora: bool = False,
        num_threads: int = 0,
    ):
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens
        self.default_params = default_params
        self.arch_config = arch_config
        self.use_lora = use_lora
        self.num_threads = num_threads
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.cached_token_ids: list[int] = []
        self._last_cache_hit: int = 0
        self._cached_prompt_str: str = ""

    async def start(self):
        """Launch the C++ inference subprocess."""
        from trillim.inference import _config_args

        if self.use_lora:
            lora_path = os.path.join(self.model_dir, "qmodel.lora")
            if not os.path.exists(lora_path):
                raise RuntimeError(
                    f"--lora flag set but {lora_path} not found. "
                    "Run 'trillim pull <model>' to download pre-built artifacts."
                )

        from trillim._bin_path import inference_bin

        cmd = [inference_bin(), self.model_dir] + _config_args(self.arch_config, num_threads=self.num_threads)
        if self.use_lora:
            cmd.append("--lora")
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stop(self):
        """Shut down the subprocess gracefully."""
        self.cached_token_ids = []
        self._last_cache_hit = 0
        self._cached_prompt_str = ""
        if self.process and self.process.returncode is None:
            self.process.stdin.write(b"0\n")
            await self.process.stdin.drain()
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
            else:
                # Token-level prefix matching (fallback for completions / no template)
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

            # Protocol: num_tokens, reset_flag, then sampling params, then token IDs
            header = f"{len(delta_tokens)}\n{reset_flag}\n{temp}\n{tk}\n{tp}\n{rp}\n{rl}\n{mt}\n"
            try:
                proc.stdin.write(header.encode())
                for tid in delta_tokens:
                    proc.stdin.write(f"{tid}\n".encode())
                await proc.stdin.drain()
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
                line = await proc.stdout.readline()
                if not line:
                    break
                token_id = int(line.strip())
                generated_tokens.append(token_id)
                if token_id in self.stop_tokens:
                    break
                yield token_id

            # Read kv_position and update cache state
            kv_line = await proc.stdout.readline()
            if kv_line:
                kv_position = int(kv_line.strip())
                self.cached_token_ids = (all_token_ids + generated_tokens)[:kv_position]
            else:
                self.cached_token_ids = []
                self._cached_prompt_str = ""


# ---------------------------------------------------------------------------
# LLM component
# ---------------------------------------------------------------------------


class LLM(Component):
    """CPU inference component — manages the C++ subprocess and exposes
    /v1/models, /v1/models/load, /v1/chat/completions, /v1/completions."""

    def __init__(self, model_dir: str, num_threads: int = 0, trust_remote_code: bool = False):
        self._model_dir = model_dir
        self._use_lora = False
        self._num_threads = num_threads
        self._trust_remote_code = trust_remote_code
        self.engine: InferenceEngine | None = None
        self.model_name: str = "unknown"
        self.state: ServerState = ServerState.NO_MODEL
        self._swap_lock: asyncio.Lock | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        from trillim.model_arch import ModelConfig as ArchConfig
        from trillim.inference import load_tokenizer

        self._swap_lock = asyncio.Lock()

        self.model_name = os.path.basename(os.path.normpath(self._model_dir))
        config_path = os.path.join(self._model_dir, "config.json")

        tokenizer = load_tokenizer(self._model_dir, self._use_lora, trust_remote_code=self._trust_remote_code)
        arch_config = ArchConfig.from_config_json(config_path, self._model_dir)
        stop_tokens = set(arch_config.eos_tokens)
        default_params = load_default_params(self._model_dir)

        self.engine = InferenceEngine(
            self._model_dir,
            tokenizer,
            stop_tokens,
            default_params,
            arch_config=arch_config,
            use_lora=self._use_lora,
            num_threads=self._num_threads,
        )
        await self.engine.start()
        self.state = ServerState.RUNNING

    async def stop(self) -> None:
        self.state = ServerState.NO_MODEL
        if self.engine is not None:
            await self.engine.stop()

    # -- hot-swap ------------------------------------------------------------

    async def _swap_engine(
        self,
        model_dir: str,
        adapter_dir: str | None = None,
        use_lora: bool | None = None,
        num_threads: int | None = None,
    ) -> LoadModelResponse:
        from trillim.model_arch import ModelConfig as ArchConfig
        from trillim.inference import load_tokenizer

        # Resolve LoRA setting: explicit > adapter_dir implies true > current default
        if use_lora is not None:
            lora = use_lora
        elif adapter_dir is not None:
            lora = True
        else:
            lora = False

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            return LoadModelResponse(
                status="error",
                model=self.model_name,
                recompiled=False,
                message=f"config.json not found in {model_dir}",
            )

        # Stop the old engine
        if self.engine is not None:
            async with self.engine.lock:
                await self.engine.stop()
            self.engine = None

        # Validate LoRA file exists if requested
        lora_path = os.path.join(model_dir, "qmodel.lora")
        if lora and not os.path.exists(lora_path):
            self.state = ServerState.NO_MODEL
            return LoadModelResponse(
                status="error",
                model=self.model_name,
                recompiled=False,
                message=f"LoRA requested but {lora_path} not found. "
                "Run 'make quantize MODEL_DIR=<path> ADAPTER_DIR=<path>' first.",
            )

        # Load new tokenizer, config, and params
        try:
            tokenizer = load_tokenizer(model_dir, lora, trust_remote_code=self._trust_remote_code)
            arch_config = ArchConfig.from_config_json(config_path, model_dir)
            stop_tokens = set(arch_config.eos_tokens)
            default_params = load_default_params(model_dir)
        except Exception as exc:
            self.state = ServerState.NO_MODEL
            return LoadModelResponse(
                status="error",
                model=self.model_name,
                recompiled=False,
                message=f"Failed to load model config: {exc}",
            )

        # Start new engine
        threads = num_threads if num_threads is not None else self._num_threads
        new_name = os.path.basename(os.path.normpath(model_dir))
        new_engine = InferenceEngine(
            model_dir,
            tokenizer,
            stop_tokens,
            default_params,
            arch_config=arch_config,
            use_lora=lora,
            num_threads=threads,
        )
        try:
            await new_engine.start()
        except Exception as exc:
            self.state = ServerState.NO_MODEL
            return LoadModelResponse(
                status="error",
                model=new_name,
                recompiled=False,
                message=f"Failed to start engine: {exc}",
            )

        self.engine = new_engine
        self._use_lora = lora
        self._num_threads = threads
        self.model_name = new_name
        self.state = ServerState.RUNNING

        return LoadModelResponse(
            status="success",
            model=self.model_name,
            recompiled=False,
        )

    # -- router --------------------------------------------------------------

    def router(self) -> APIRouter:
        r = APIRouter()
        llm = self  # closure reference

        @r.get("/v1/models")
        async def list_models():
            if llm.state != ServerState.RUNNING:
                return ModelListResponse(data=[])
            return ModelListResponse(data=[ModelInfo(id=llm.model_name)])

        @r.post("/v1/models/load")
        async def load_model(req: LoadModelRequest):
            from trillim.model_store import resolve_model_dir, MODELS_DIR

            try:
                model_dir = resolve_model_dir(req.model_dir)
            except SystemExit:
                raise HTTPException(status_code=404, detail=f"Model '{req.model_dir}' not found")
            resolved = os.path.realpath(model_dir)
            allowed = os.path.realpath(str(MODELS_DIR))
            if not resolved.startswith(allowed + os.sep):
                raise HTTPException(
                    status_code=403,
                    detail="Only models in ~/.trillim/models/ can be loaded. Use 'trillim pull' first.",
                )

            if llm._swap_lock is not None and llm._swap_lock.locked():
                raise HTTPException(
                    status_code=409,
                    detail="Model swap already in progress",
                )

            async with llm._swap_lock:
                llm.state = ServerState.SWAPPING
                result = await llm._swap_engine(
                    model_dir,
                    adapter_dir=req.adapter_dir,
                    use_lora=req.lora,
                    num_threads=req.threads,
                )

            if result.status == "error":
                if llm.engine is not None:
                    llm.state = ServerState.RUNNING
                raise HTTPException(status_code=500, detail=result.message)
            return result

        @r.post("/v1/chat/completions")
        async def chat_completions(req: ChatCompletionRequest):
            if llm.state == ServerState.SWAPPING:
                raise HTTPException(status_code=503, detail="Model swap in progress")
            if llm.engine is None or llm.state != ServerState.RUNNING:
                raise HTTPException(status_code=503, detail="No model loaded")

            tokenizer = llm.engine.tokenizer

            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            has_chat_template = (
                hasattr(tokenizer, "chat_template") and tokenizer.chat_template
            )
            if has_chat_template:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            else:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                prompt += "\nassistant:"
                token_ids = tokenizer.encode(prompt)

            max_ctx = llm.engine.arch_config.max_position_embeddings
            if len(token_ids) >= max_ctx:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt length ({len(token_ids)} tokens) exceeds context window ({max_ctx}). "
                    "Start a new conversation with fewer messages.",
                )

            gen_kwargs = dict(
                token_ids=token_ids,
                prompt_str=prompt if has_chat_template else None,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
            )

            if req.stream:
                return StreamingResponse(
                    _stream_chat(
                        llm, gen_kwargs, req.model or llm.model_name, messages
                    ),
                    media_type="text/event-stream",
                )

            from trillim.token_utils import IncrementalDecoder

            decoder = IncrementalDecoder(tokenizer)
            full_text = ""
            completion_tokens = 0
            async for token_id in llm.engine.generate(**gen_kwargs):
                full_text += decoder.decode(token_id)
                completion_tokens += 1

            cached_tokens = llm.engine._last_cache_hit

            # Update cached prompt string for next turn's string-level matching
            if has_chat_template:
                post_messages = messages + [{"role": "assistant", "content": full_text}]
                llm.engine._cached_prompt_str = tokenizer.apply_chat_template(
                    post_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

            return ChatCompletionResponse(
                id=make_id(),
                created=now(),
                model=req.model or llm.model_name,
                choices=[
                    ChatChoice(
                        message=ChatMessage(role="assistant", content=full_text),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=len(token_ids),
                    completion_tokens=completion_tokens,
                    total_tokens=len(token_ids) + completion_tokens,
                    cached_tokens=cached_tokens,
                ),
            )

        @r.post("/v1/completions")
        async def completions(req: CompletionRequest):
            if llm.state == ServerState.SWAPPING:
                raise HTTPException(status_code=503, detail="Model swap in progress")
            if llm.engine is None or llm.state != ServerState.RUNNING:
                raise HTTPException(status_code=503, detail="No model loaded")

            tokenizer = llm.engine.tokenizer
            token_ids = tokenizer.encode(req.prompt)

            max_ctx = llm.engine.arch_config.max_position_embeddings
            if len(token_ids) >= max_ctx:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt length ({len(token_ids)} tokens) exceeds context window ({max_ctx}). "
                    "Shorten your prompt and try again.",
                )

            gen_kwargs = dict(
                token_ids=token_ids,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
            )

            if req.stream:
                return StreamingResponse(
                    _stream_completion(
                        llm, gen_kwargs, req.model or llm.model_name
                    ),
                    media_type="text/event-stream",
                )

            from trillim.token_utils import IncrementalDecoder

            decoder = IncrementalDecoder(tokenizer)
            full_text = ""
            completion_tokens = 0
            async for token_id in llm.engine.generate(**gen_kwargs):
                full_text += decoder.decode(token_id)
                completion_tokens += 1

            cached_tokens = llm.engine._last_cache_hit

            return CompletionResponse(
                id=make_id(),
                created=now(),
                model=req.model or llm.model_name,
                choices=[CompletionChoice(text=full_text, finish_reason="stop")],
                usage=UsageInfo(
                    prompt_tokens=len(token_ids),
                    completion_tokens=completion_tokens,
                    total_tokens=len(token_ids) + completion_tokens,
                    cached_tokens=cached_tokens,
                ),
            )

        return r


# ---------------------------------------------------------------------------
# Streaming helpers (module-level async generators)
# ---------------------------------------------------------------------------


async def _stream_chat(
    llm: LLM, gen_kwargs: dict, model: str, messages: list[dict] | None = None
):
    from trillim.token_utils import IncrementalDecoder

    req_id = make_id()
    created = now()

    chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    tokenizer = llm.engine.tokenizer
    decoder = IncrementalDecoder(tokenizer)
    full_text = ""
    async for token_id in llm.engine.generate(**gen_kwargs):
        text = decoder.decode(token_id)
        full_text += text
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Update cached prompt string for next turn's string-level matching
    if (
        messages is not None
        and hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template
    ):
        post_messages = messages + [{"role": "assistant", "content": full_text}]
        llm.engine._cached_prompt_str = tokenizer.apply_chat_template(
            post_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_completion(llm: LLM, gen_kwargs: dict, model: str):
    from trillim.token_utils import IncrementalDecoder

    req_id = make_id()
    created = now()

    decoder = IncrementalDecoder(llm.engine.tokenizer)
    async for token_id in llm.engine.generate(**gen_kwargs):
        text = decoder.decode(token_id)
        chunk = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    chunk = {
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
