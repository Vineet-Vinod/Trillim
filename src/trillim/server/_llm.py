# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""LLM component — wraps InferenceEngine and exposes inference routes."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from trillim.utils import load_default_params

from ._component import Component

from ._helpers import make_id, now
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

from trillim.errors import ContextOverflowError
from trillim._prompt_cache import PromptSnapshot
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

_APPEND_TOKEN_VALIDATION_TAIL_CHARS = (8, 16, 32)


class ChatSession:
    """Append-only multi-turn chat state for a single active model.

    Prompt rendering must stay append-only. Incremental suffix tokenization is
    only a fast path: ChatSession validates a small overlap near the append
    boundary and falls back to full prompt re-encoding when a tokenizer needs
    more left-context than that window can prove.
    """

    _runtime_proxy = True

    def __init__(
        self,
        llm: LLM,
        messages: list[dict] | None = None,
    ):
        self._llm = llm
        self._llm_generation = llm._session_generation
        self._messages: list[dict] = []
        self._base_prompt_str = ""
        self._base_token_ids: list[int] = []
        self._prepared_prompt_str: str | None = None
        self._prepared_token_ids: list[int] | None = None
        for message in messages or []:
            self._append_message(message["role"], message["content"])

    def _require_active(self) -> tuple[InferenceEngine, Harness]:
        engine, harness = self._llm._require_started()
        if self._llm_generation != self._llm._session_generation:
            raise RuntimeError(
                "ChatSession is stale after the active model changed; create a new session"
            )
        return engine, harness

    def _render_prompt(self, *, add_generation_prompt: bool) -> str:
        engine, _ = self._require_active()
        tokenizer = engine.tokenizer
        has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        if has_template:
            return tokenizer.apply_chat_template(
                self._messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in self._messages)
        if add_generation_prompt:
            if prompt:
                return f"{prompt}\nassistant: "
            return "assistant: "
        return prompt

    def _encode_full_prompt(self, prompt_str: str) -> list[int]:
        engine, _ = self._require_active()
        tokenizer = engine.tokenizer
        has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        return tokenizer.encode(prompt_str, add_special_tokens=not has_template)

    def _encode_suffix(self, suffix: str) -> list[int]:
        if not suffix:
            return []
        engine, _ = self._require_active()
        return engine.tokenizer.encode(suffix, add_special_tokens=False)

    def _suffix_passes_overlap_validation(
        self,
        base_prompt_str: str,
        suffix: str,
        suffix_token_ids: list[int],
    ) -> bool:
        if not base_prompt_str or not suffix:
            return True
        engine, _ = self._require_active()
        tokenizer = engine.tokenizer
        checked_tail_lengths: set[int] = set()
        for tail_chars in _APPEND_TOKEN_VALIDATION_TAIL_CHARS:
            tail_len = min(len(base_prompt_str), tail_chars)
            if tail_len in checked_tail_lengths:
                continue
            checked_tail_lengths.add(tail_len)
            tail_text = base_prompt_str[-tail_len:]
            tail_token_ids = tokenizer.encode(tail_text, add_special_tokens=False)
            combined_token_ids = tokenizer.encode(
                tail_text + suffix,
                add_special_tokens=False,
            )
            if combined_token_ids != tail_token_ids + suffix_token_ids:
                return False
        return True

    def _materialize_append_only_tokens(
        self,
        prompt_str: str,
        *,
        base_prompt_str: str,
        base_token_ids: list[int],
        context: str,
    ) -> list[int]:
        """Return exact prompt tokens for an append-only prompt update.

        Suffix-only encoding is the fast path. Tokenizers that merge across a
        longer boundary than the local validation window lose that fast path and
        pay a full prompt re-encode here instead.
        """
        if not base_prompt_str and not base_token_ids:
            return self._encode_full_prompt(prompt_str)
        if not prompt_str.startswith(base_prompt_str):
            raise RuntimeError(
                f"ChatSession requires append-only prompt rendering; {context} rewrote earlier prompt content"
            )
        suffix = prompt_str[len(base_prompt_str):]
        if not suffix:
            return list(base_token_ids)
        suffix_token_ids = self._encode_suffix(suffix)
        if self._suffix_passes_overlap_validation(
            base_prompt_str,
            suffix,
            suffix_token_ids,
        ):
            return list(base_token_ids) + suffix_token_ids
        return self._encode_full_prompt(prompt_str)

    def _require_turn_ready(self) -> None:
        self._require_active()
        if not self._messages:
            raise ValueError("ChatSession has no messages")
        if self._messages[-1]["role"] == "assistant":
            raise ValueError(
                "ChatSession already has an assistant reply; append a new message before chatting again"
            )

    def _append_message(self, role: str, content: str) -> None:
        self._require_active()
        self._prepared_prompt_str = None
        self._prepared_token_ids = None
        self._messages.append({"role": role, "content": content})
        prompt_str = self._render_prompt(add_generation_prompt=False)
        self._base_token_ids = self._materialize_append_only_tokens(
            prompt_str,
            base_prompt_str=self._base_prompt_str,
            base_token_ids=self._base_token_ids,
            context=f"appending {role!r} message",
        )
        self._base_prompt_str = prompt_str

    def _prepare_reply(self) -> tuple[list[int], str]:
        self._require_turn_ready()
        if self._prepared_token_ids is not None and self._prepared_prompt_str is not None:
            return list(self._prepared_token_ids), self._prepared_prompt_str
        prompt_str = self._render_prompt(add_generation_prompt=True)
        self._prepared_token_ids = self._materialize_append_only_tokens(
            prompt_str,
            base_prompt_str=self._base_prompt_str,
            base_token_ids=self._base_token_ids,
            context="preparing the next assistant turn",
        )
        self._prepared_prompt_str = prompt_str
        return list(self._prepared_token_ids), self._prepared_prompt_str

    def _finalize_assistant(self, text: str, token_ids: list[int]) -> PromptSnapshot:
        self._require_active()
        if self._prepared_prompt_str is None or self._prepared_token_ids is None:
            raise RuntimeError("ChatSession assistant turn was not prepared")
        self._messages.append({"role": "assistant", "content": text})
        prompt_str = self._render_prompt(add_generation_prompt=False)
        generated_prefix = self._prepared_prompt_str + text
        cache_snapshot = PromptSnapshot.create(
            list(self._prepared_token_ids) + list(token_ids),
            generated_prefix,
        )
        self._base_token_ids = self._materialize_append_only_tokens(
            prompt_str,
            base_prompt_str=generated_prefix,
            base_token_ids=list(self._prepared_token_ids) + list(token_ids),
            context="finalizing assistant output",
        )
        self._base_prompt_str = prompt_str
        self._prepared_prompt_str = None
        self._prepared_token_ids = None
        return cache_snapshot

    @property
    def messages(self) -> tuple[dict, ...]:
        self._require_active()
        return tuple(
            {"role": message["role"], "content": message["content"]}
            for message in self._messages
        )

    @property
    def max_context_tokens(self) -> int:
        engine, _ = self._require_active()
        return engine.arch_config.max_position_embeddings

    @property
    def prompt_tokens(self) -> int:
        token_ids, _ = self._prepare_reply()
        return len(token_ids)

    @property
    def remaining_context_tokens(self) -> int:
        return self.max_context_tokens - self.prompt_tokens

    def validate(self) -> int:
        self._require_turn_ready()
        token_count = self.prompt_tokens
        if token_count >= self.max_context_tokens:
            raise ContextOverflowError(token_count, self.max_context_tokens)
        return token_count

    def add_user(self, content: str) -> None:
        self._append_message("user", content)

    def add_system(self, content: str) -> None:
        self._append_message("system", content)

    async def stream_chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[ChatEvent, None]:
        engine, harness = self._require_active()
        prompt_tokens = self.validate()
        full_text = ""

        async for event in harness.stream_events(
            self,
            **self._llm._chat_sampling(
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
            cached_tokens=engine.last_cache_hit,
        )
        yield ChatDoneEvent(text=full_text, usage=usage)

    async def _collect_chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> tuple[str, ChatUsage]:
        full_text = ""
        usage: ChatUsage | None = None

        events = self.stream_chat(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

        try:
            while True:
                try:
                    if timeout is None:
                        event = await events.__anext__()
                    else:
                        event = await asyncio.wait_for(
                            events.__anext__(),
                            timeout=timeout,
                        )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"LLM chat timed out after {timeout} seconds."
                    ) from exc

                if isinstance(event, ChatTokenEvent):
                    full_text += event.text
                elif isinstance(event, ChatFinalTextEvent):
                    full_text = event.text
                elif isinstance(event, ChatDoneEvent):
                    full_text = event.text
                    usage = event.usage
        finally:
            await events.aclose()

        if usage is None:
            raise RuntimeError("Chat stream ended without a done event")
        return full_text, usage

    async def chat(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        try:
            full_text, _ = await self._collect_chat(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except TimeoutError:
            await self._llm._restart_after_timeout("LLM chat")
            raise
        return full_text


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
        self._session_generation = 0

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

    def _validate_token_count(self, token_count: int) -> int:
        max_context_tokens = self.max_context_tokens
        if token_count >= max_context_tokens:
            raise ContextOverflowError(token_count, max_context_tokens)
        return token_count

    def _clone_messages(self, messages: list[dict]) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def session(self, messages: list[dict] | None = None) -> ChatSession:
        self._require_started()
        return ChatSession(self, self._clone_messages(messages or []))

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
        session = self.session(messages)
        async for event in session.stream_chat(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        ):
            yield event

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
        return await self.session(messages)._collect_chat(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )

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
        return await self.session(messages).chat(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def _restart_after_timeout(self, operation: str) -> None:
        if self._swap_lock is None:
            self._swap_lock = asyncio.Lock()

        async with self._swap_lock:
            self.state = ServerState.SWAPPING
            result = await self._swap_engine(
                self._model_dir,
                adapter_dir=self._adapter_dir,
                harness_name=self._harness_name,
                search_provider=self._search_provider,
                num_threads=self._num_threads,
                lora_quant=self._lora_quant,
                unembed_quant=self._unembed_quant,
            )

        if result.status == "error":
            raise RuntimeError(
                f"{operation} timed out and engine restart failed: {result.message}"
            )

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
        self._session_generation += 1

    async def stop(self) -> None:
        self.state = ServerState.NO_MODEL
        if self.engine is not None:
            await self.engine.stop()
        self.harness = None
        self.engine = None
        self._session_generation += 1

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
        self._session_generation += 1

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
            from pathlib import Path

            allowed = Path(str(MODELS_DIR)).resolve()

            def resolve_allowed_dir(path_arg: str, *, kind: str) -> str:
                try:
                    resolved_dir = resolve_model_dir(path_arg)
                except RuntimeError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc

                resolved_path = Path(resolved_dir).resolve()
                try:
                    resolved_path.relative_to(allowed)
                except ValueError as exc:
                    detail = (
                        "Only models in ~/.trillim/models/ can be loaded. Use 'trillim pull' first."
                        if kind == "model"
                        else "Only adapters in ~/.trillim/models/ can be loaded. Use 'trillim pull' first."
                    )
                    raise HTTPException(status_code=403, detail=detail) from exc
                return resolved_dir

            model_dir = resolve_allowed_dir(req.model_dir, kind="model")
            adapter_dir = None
            if req.adapter_dir is not None:
                adapter_dir = resolve_allowed_dir(req.adapter_dir, kind="adapter")

            if llm._swap_lock is not None and llm._swap_lock.locked():
                raise HTTPException(
                    status_code=409,
                    detail="Model swap already in progress",
                )

            async with llm._swap_lock:
                llm.state = ServerState.SWAPPING
                result = await llm._swap_engine(
                    model_dir,
                    adapter_dir=adapter_dir,
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
                session = llm.session(messages)
                try:
                    session.validate()
                except (ContextOverflowError, ValueError) as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{exc} Start a new conversation with fewer messages.",
                    )
                return StreamingResponse(
                    _stream_chat(
                        session, sampling, req.model or llm.model_name
                    ),
                    media_type="text/event-stream",
                )

            try:
                full_text, usage = await llm._collect_chat(messages, **sampling)
            except (ContextOverflowError, ValueError) as exc:
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

            cached_tokens = llm.engine.last_cache_hit

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


async def _stream_chat(session: ChatSession, sampling: dict, model: str):
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

    async for event in session.stream_chat(**sampling):
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
