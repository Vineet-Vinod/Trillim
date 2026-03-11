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

from trillim._timeouts import run_with_timeout
from trillim.errors import ContextOverflowError
from trillim.engine import InferenceEngine
from trillim.events import (
    ChatDoneEvent,
    ChatEvent,
    ChatFinalTextEvent,
    ChatTokenEvent,
    ChatUsage,
)
from trillim.harnesses import Harness, get_harness


# ---------------------------------------------------------------------------
# LLM component
# ---------------------------------------------------------------------------


class LLM(Component):
    """CPU inference component — manages the C++ subprocess and exposes
    /v1/models, /v1/models/load, /v1/chat/completions, /v1/completions."""

    def __init__(self, model_dir: str, adapter_dir: str | None = None, num_threads: int = 0, trust_remote_code: bool = False, lora_quant: str | None = None, unembed_quant: str | None = None, harness_name: str = "default"):
        self._model_dir = model_dir
        self._adapter_dir = adapter_dir
        self._num_threads = num_threads
        self._trust_remote_code = trust_remote_code
        self._lora_quant = lora_quant
        self._unembed_quant = unembed_quant
        self._harness_name = harness_name
        self._search_provider = "ddgs"
        self.engine: InferenceEngine | None = None
        self.harness: Harness | None = None
        self.model_name: str = "unknown"
        self.state: ServerState = ServerState.NO_MODEL
        self._swap_lock: asyncio.Lock | None = None

    def _require_started(self) -> tuple[InferenceEngine, Harness]:
        if (
            self.engine is None
            or self.harness is None
            or self.state != ServerState.RUNNING
        ):
            raise RuntimeError("LLM not started")
        return self.engine, self.harness

    @property
    def max_context_tokens(self) -> int:
        """Return the active model context window in tokens."""
        engine, _ = self._require_started()
        return engine.arch_config.max_position_embeddings

    def count_tokens(self, messages: list[dict]) -> int:
        """Count prompt tokens for a chat-style message list."""
        _, harness = self._require_started()
        token_ids, _ = harness._prepare_tokens(messages)
        return len(token_ids)

    def validate_context(self, messages: list[dict]) -> int:
        """Return prompt token count or raise if the context window is full."""
        return self._validate_token_count(self.count_tokens(messages))

    def _validate_token_count(self, token_count: int) -> int:
        max_context_tokens = self.max_context_tokens
        if token_count >= max_context_tokens:
            raise ContextOverflowError(token_count, max_context_tokens)
        return token_count

    def _clone_messages(self, messages: list[dict]) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def _chat_sampling(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        return dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        engine, harness = self._require_started()
        conversation = self._clone_messages(messages)
        prompt_tokens = self.validate_context(conversation)
        full_text = ""

        async for event in harness.stream_events(
            conversation,
            **self._chat_sampling(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            ),
        ):
            if isinstance(event, ChatTokenEvent):
                full_text += event.text
            elif isinstance(event, ChatFinalTextEvent):
                full_text = event.text
            yield event

        usage = ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=harness._last_completion_tokens,
            total_tokens=prompt_tokens + harness._last_completion_tokens,
            cached_tokens=engine._last_cache_hit,
        )
        yield ChatDoneEvent(text=full_text, usage=usage)

    async def _collect_chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, ChatUsage]:
        full_text = ""
        usage: ChatUsage | None = None

        async for event in self.stream_chat(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        ):
            if isinstance(event, ChatTokenEvent):
                full_text += event.text
            elif isinstance(event, ChatFinalTextEvent):
                full_text = event.text
            elif isinstance(event, ChatDoneEvent):
                full_text = event.text
                usage = event.usage

        if usage is None:
            raise RuntimeError("Chat stream ended without a done event")
        return full_text, usage

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        full_text, _ = await run_with_timeout(
            self._collect_chat(
                messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            ),
            timeout,
            "LLM chat",
        )
        return full_text

    def _create_harness(
        self,
        engine: InferenceEngine,
        harness_name: str,
        search_provider: str,
    ) -> Harness:
        harness_cls = get_harness(harness_name)
        if harness_name == "search":
            return harness_cls(engine, search_provider=search_provider)
        return harness_cls(engine)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        from trillim.model_arch import ModelConfig as ArchConfig
        from trillim.utils import load_tokenizer

        self._swap_lock = asyncio.Lock()

        self.model_name = os.path.basename(os.path.normpath(self._model_dir))
        config_path = os.path.join(self._model_dir, "config.json")

        tokenizer = load_tokenizer(self._model_dir, adapter_dir=self._adapter_dir, trust_remote_code=self._trust_remote_code)
        arch_config = ArchConfig.from_config_json(config_path, self._model_dir, adapter_dir=self._adapter_dir)
        stop_tokens = set(arch_config.eos_tokens)
        default_params = load_default_params(self._model_dir)

        self.engine = InferenceEngine(
            self._model_dir,
            tokenizer,
            stop_tokens,
            default_params,
            arch_config=arch_config,
            adapter_dir=self._adapter_dir,
            num_threads=self._num_threads,
            lora_quant=self._lora_quant,
            unembed_quant=self._unembed_quant,
        )
        await self.engine.start()
        self.harness = self._create_harness(
            self.engine, self._harness_name, self._search_provider
        )
        self.state = ServerState.RUNNING

    async def stop(self) -> None:
        self.state = ServerState.NO_MODEL
        if self.engine is not None:
            await self.engine.stop()
        self.harness = None

    # -- hot-swap ------------------------------------------------------------

    async def _swap_engine(
        self,
        model_dir: str,
        adapter_dir: str | None = None,
        harness_name: str | None = None,
        search_provider: str | None = None,
        num_threads: int | None = None,
        lora_quant: str | None = None,
        unembed_quant: str | None = None,
    ) -> LoadModelResponse:
        from trillim.model_arch import ModelConfig as ArchConfig
        from trillim.utils import load_tokenizer
        from trillim.model_store import resolve_model_dir

        # Resolve adapter_dir if provided
        resolved_adapter: str | None = None
        if adapter_dir is not None:
            resolved_adapter = resolve_model_dir(adapter_dir)

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            return LoadModelResponse(
                status="error",
                model=self.model_name,
                recompiled=False,
                message=f"config.json not found in {model_dir}",
            )

        next_harness = harness_name if harness_name is not None else self._harness_name
        next_search_provider = (
            search_provider
            if search_provider is not None
            else self._search_provider
        )

        # Validate harness/provider before stopping the old engine.
        if self.engine is not None:
            try:
                self._create_harness(self.engine, next_harness, next_search_provider)
            except Exception as exc:
                return LoadModelResponse(
                    status="error",
                    model=self.model_name,
                    recompiled=False,
                    message=f"Invalid harness config: {exc}",
                )

        # Validate LoRA adapter before stopping the old engine
        if resolved_adapter:
            trillim_cfg_path = os.path.join(resolved_adapter, "trillim_config.json")
            if not os.path.exists(trillim_cfg_path):
                return LoadModelResponse(
                    status="error",
                    model=self.model_name,
                    recompiled=False,
                    message=f"{trillim_cfg_path} not found. "
                    "This adapter has not been quantized for Trillim. "
                    f"Run: trillim quantize <model_dir> --adapter {resolved_adapter}",
                )
            lora_path = os.path.join(resolved_adapter, "qmodel.lora")
            if not os.path.exists(lora_path):
                return LoadModelResponse(
                    status="error",
                    model=self.model_name,
                    recompiled=False,
                    message=f"LoRA requested but {lora_path} not found. "
                    f"Run: trillim quantize <model_dir> --adapter {resolved_adapter}",
                )
            from trillim.model_store import AdapterCompatError, validate_adapter_model_compat
            try:
                validate_adapter_model_compat(resolved_adapter, model_dir)
            except AdapterCompatError as e:
                return LoadModelResponse(
                    status="error",
                    model=self.model_name,
                    recompiled=False,
                    message=str(e),
                )

        # Load new tokenizer, config, and params before stopping the old engine
        # so a failure here doesn't leave the server with no model running
        try:
            tokenizer = load_tokenizer(model_dir, adapter_dir=resolved_adapter, trust_remote_code=self._trust_remote_code)
            arch_config = ArchConfig.from_config_json(config_path, model_dir, adapter_dir=resolved_adapter)
            stop_tokens = set(arch_config.eos_tokens)
            default_params = load_default_params(model_dir)
        except Exception as exc:
            return LoadModelResponse(
                status="error",
                model=self.model_name,
                recompiled=False,
                message=f"Failed to load model config: {exc}",
            )

        # Stop the old engine only after everything above succeeded
        if self.engine is not None:
            async with self.engine.lock:
                await self.engine.stop()
            self.engine = None

        # Start new engine
        threads = num_threads if num_threads is not None else self._num_threads
        new_name = os.path.basename(os.path.normpath(model_dir))
        new_engine = InferenceEngine(
            model_dir,
            tokenizer,
            stop_tokens,
            default_params,
            arch_config=arch_config,
            adapter_dir=resolved_adapter,
            num_threads=threads,
            lora_quant=lora_quant,
            unembed_quant=unembed_quant,
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
        try:
            self.harness = self._create_harness(
                new_engine, next_harness, next_search_provider
            )
        except Exception as exc:
            await new_engine.stop()
            self.engine = None
            self.state = ServerState.NO_MODEL
            return LoadModelResponse(
                status="error",
                model=new_name,
                recompiled=False,
                message=f"Invalid harness config: {exc}",
            )

        self._harness_name = next_harness
        self._search_provider = next_search_provider
        self._adapter_dir = resolved_adapter
        self._num_threads = threads
        self._lora_quant = lora_quant
        self._unembed_quant = unembed_quant
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
            except RuntimeError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            from pathlib import Path

            resolved = Path(model_dir).resolve()
            allowed = Path(str(MODELS_DIR)).resolve()
            try:
                resolved.relative_to(allowed)
            except ValueError:
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
                    harness_name=req.harness,
                    search_provider=req.search_provider,
                    num_threads=req.threads,
                    lora_quant=req.lora_quant,
                    unembed_quant=req.unembed_quant,
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
            if llm.engine is None or llm.harness is None or llm.state != ServerState.RUNNING:
                raise HTTPException(status_code=503, detail="No model loaded")

            messages = [{"role": m.role, "content": m.content} for m in req.messages]

            sampling = dict(
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_tokens=req.max_tokens,
            )

            if req.stream:
                try:
                    llm.validate_context(messages)
                except ContextOverflowError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{exc} Start a new conversation with fewer messages.",
                    )
                return StreamingResponse(
                    _stream_chat(
                        llm, messages, sampling, req.model or llm.model_name
                    ),
                    media_type="text/event-stream",
                )

            try:
                full_text, usage = await llm._collect_chat(messages, **sampling)
            except ContextOverflowError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"{exc} Start a new conversation with fewer messages.",
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
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cached_tokens=usage.cached_tokens,
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
            try:
                prompt_tokens = llm._validate_token_count(len(token_ids))
            except ContextOverflowError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"{exc} Shorten your prompt and try again.",
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
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    cached_tokens=cached_tokens,
                ),
            )

        return r


# ---------------------------------------------------------------------------
# Streaming helpers (module-level async generators)
# ---------------------------------------------------------------------------


async def _stream_chat(llm: LLM, messages: list[dict], sampling: dict, model: str):
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

    async for event in llm.stream_chat(messages, **sampling):
        if not isinstance(event, ChatTokenEvent):
            continue
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": event.text}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

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
