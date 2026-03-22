"""Public LLM component and chat session API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import weakref
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from fastapi import APIRouter

from trillim.components import Component
from trillim.components.llm._admission import GenerationAdmission
from trillim.components.llm._config import LLMState, ModelInfo, load_sampling_defaults
from trillim.components.llm._engine import InferenceEngine
from trillim.components.llm._limits import MAX_THREADS, TOKEN_PROGRESS_TIMEOUT_SECONDS
from trillim.components.llm._model_dir import validate_model_dir
from trillim.components.llm._router import build_router
from trillim.components.llm._session import ChatSession, _ChatSession, _create_chat_session
from trillim.components.llm._swap import _wait_for_idle_or_cancel, restart_model, swap_model
from trillim.components.llm._tokenizer import load_tokenizer
from trillim.components.llm._validation import validate_messages
from trillim.harnesses._default import _DefaultHarness
from trillim.harnesses.search._harness import _SearchHarness
from trillim.harnesses.search.provider import (
    DEFAULT_SEARCH_TOKEN_BUDGET,
    normalize_provider_name,
    resolve_search_token_budget,
    validate_harness_name,
)
from trillim.errors import AdmissionRejectedError, ComponentLifecycleError


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    harness_name: str
    search_provider: str
    search_token_budget: int
    requested_search_token_budget: int


class LLM(Component):
    """LLM component with truthful model state and bounded chat sessions."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        num_threads: int = 0,
        trust_remote_code: bool = False,
        harness_name: str = "default",
        search_provider: str = "ddgs",
        search_token_budget: int = DEFAULT_SEARCH_TOKEN_BUDGET,
        _model_validator=validate_model_dir,
        _tokenizer_loader=load_tokenizer,
        _engine_factory=InferenceEngine,
    ) -> None:
        """Configure an LLM component for one model directory."""
        if not model_dir:
            raise ValueError("model_dir is required")
        if num_threads < 0 or num_threads > MAX_THREADS:
            raise ValueError(f"num_threads must be between 0 and {MAX_THREADS}")
        if search_token_budget < 1:
            raise ValueError("search_token_budget must be at least 1")
        self._configured_model_dir = str(model_dir)
        self._num_threads = num_threads
        self._trust_remote_code = trust_remote_code
        self._configured_harness_name = validate_harness_name(harness_name)
        self._configured_search_provider = normalize_provider_name(search_provider)
        self._configured_search_token_budget = search_token_budget
        self._model_validator = _model_validator
        self._tokenizer_loader = _tokenizer_loader
        self._engine_factory = _engine_factory
        self._runtime_model = None
        self._tokenizer = None
        self._defaults = None
        self._engine = None
        self._harness = None
        self._runtime_search_token_budget: int | None = None
        self._state = LLMState.UNAVAILABLE
        self._admission = GenerationAdmission()
        self._swap_lock = asyncio.Lock()
        self._sessions: weakref.WeakSet[_ChatSession] = weakref.WeakSet()
        self._hot_swap_routes_enabled = False

    @property
    def state(self) -> LLMState:
        """Return the current LLM runtime state."""
        return self._state

    @property
    def model_name(self) -> str | None:
        """Return the active model name, if any."""
        return None if self._runtime_model is None else self._runtime_model.name

    @property
    def max_context_tokens(self) -> int:
        """Return the active model context window."""
        if self._runtime_model is None:
            raise RuntimeError("LLM not started")
        return self._runtime_model.max_position_embeddings

    def router(self) -> APIRouter:
        """Return the FastAPI router for this LLM component."""
        return build_router(self, allow_hot_swap=self._hot_swap_routes_enabled)

    async def start(self) -> None:
        """Load the configured model and start the inference engine."""
        if self._engine is not None and self._state == LLMState.RUNNING:
            return
        try:
            validated, tokenizer, defaults, engine, runtime_options = self._build_runtime(
                self._configured_model_dir,
                harness_name=self._configured_harness_name,
                search_provider=self._configured_search_provider,
                search_token_budget=self._configured_search_token_budget,
            )
            await engine.start()
        except Exception:
            self._clear_runtime()
            self._state = LLMState.SERVER_ERROR
            raise
        self._bind_runtime(
            validated,
            tokenizer,
            defaults,
            engine,
            harness_name=runtime_options.harness_name,
            search_provider=runtime_options.search_provider,
            search_token_budget=runtime_options.search_token_budget,
        )
        self._state = LLMState.RUNNING
        await self._admission.finish_swapping()

    async def stop(self) -> None:
        """Stop the inference engine and invalidate live sessions."""
        await self._admission.start_draining()
        sessions = list(self._sessions)
        for session in sessions:
            session._mark_owner_stopped()
        for session in sessions:
            await session._wait_for_termination()
        engine = self._engine
        self._clear_runtime()
        try:
            if engine is not None:
                await engine.stop()
        except Exception:
            self._state = LLMState.SERVER_ERROR
            raise
        self._state = LLMState.UNAVAILABLE

    def model_info(self) -> ModelInfo:
        """Return truthful runtime metadata for the active model."""
        if self._runtime_model is None:
            return ModelInfo(
                state=self._state,
                name=None,
                path=None,
                max_context_tokens=None,
                trust_remote_code=self._trust_remote_code,
            )
        return ModelInfo(
            state=self._state,
            name=self._runtime_model.name,
            path=str(self._runtime_model.path),
            max_context_tokens=self._runtime_model.max_position_embeddings,
            trust_remote_code=self._trust_remote_code,
        )

    def open_session(
        self,
        messages: Sequence[dict[str, str]] | None = None,
    ) -> ChatSession:
        """Create a new owner-managed chat session."""
        self._require_running_runtime()
        validated = validate_messages(
            messages or (),
            require_user_turn=False,
            allow_empty=True,
        )
        session = _create_chat_session(self, validated)
        self._sessions.add(session)
        return session

    async def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator:
        """Stream one assistant turn from a temporary chat session."""
        async with self.open_session(messages) as session:
            async for event in session.stream_chat(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            ):
                yield event

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Collect one assistant turn from a temporary chat session."""
        text, _usage = await self._collect_chat(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
        )
        return text

    async def swap_model(
        self,
        model_dir: str | Path,
        *,
        harness_name: str | None = None,
        search_provider: str | None = None,
        search_token_budget: int | None = None,
    ) -> ModelInfo:
        """Hot-swap to another model without restarting the server."""
        if self._state not in {LLMState.RUNNING, LLMState.SERVER_ERROR}:
            raise ComponentLifecycleError("LLM hot swap requires the component to be running")
        await swap_model(
            self,
            str(model_dir),
            harness_name=harness_name,
            search_provider=search_provider,
            search_token_budget=search_token_budget,
        )
        return self.model_info()

    def _set_hot_swap_routes_enabled(self, enabled: bool) -> None:
        self._hot_swap_routes_enabled = enabled

    async def _collect_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ):
        async with self.open_session(messages) as session:
            text = ""
            usage = None
            async for event in session.stream_chat(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            ):
                if hasattr(event, "usage"):
                    usage = event.usage
                    text = event.text
                elif hasattr(event, "text"):
                    text = event.text
            if usage is None:
                raise RuntimeError("Chat stream ended without a done event")
            return text, usage

    def _build_runtime(
        self,
        model_dir: str | Path,
        *,
        harness_name: str | None = None,
        search_provider: str | None = None,
        search_token_budget: int | None = None,
    ):
        validated = self._model_validator(model_dir)
        tokenizer = self._tokenizer_loader(
            validated.path,
            trust_remote_code=self._trust_remote_code,
        )
        defaults = load_sampling_defaults(validated.path)
        engine = self._engine_factory(
            validated,
            tokenizer,
            defaults,
            num_threads=self._num_threads,
            progress_timeout=TOKEN_PROGRESS_TIMEOUT_SECONDS,
        )
        runtime_options = _RuntimeOptions(
            harness_name=validate_harness_name(
                self._configured_harness_name if harness_name is None else harness_name
            ),
            search_provider=normalize_provider_name(
                self._configured_search_provider
                if search_provider is None
                else search_provider
            ),
            search_token_budget=resolve_search_token_budget(
                self._configured_search_token_budget
                if search_token_budget is None
                else search_token_budget,
                max_context_tokens=validated.max_position_embeddings,
            ),
            requested_search_token_budget=(
                self._configured_search_token_budget
                if search_token_budget is None
                else search_token_budget
            ),
        )
        return validated, tokenizer, defaults, engine, runtime_options

    def _bind_runtime(
        self,
        validated,
        tokenizer,
        defaults,
        engine,
        *,
        harness_name: str,
        search_provider: str,
        search_token_budget: int,
    ) -> None:
        self._runtime_model = validated
        self._tokenizer = tokenizer
        self._defaults = defaults
        self._engine = engine
        self._runtime_search_token_budget = search_token_budget
        if harness_name == "default":
            self._harness = _DefaultHarness(engine)
            return
        self._harness = _SearchHarness(
            engine,
            search_provider=search_provider,
            search_token_budget=search_token_budget,
        )

    def _update_configured_runtime(
        self,
        *,
        model_dir: str,
        harness_name: str,
        search_provider: str,
        search_token_budget: int,
    ) -> None:
        self._configured_model_dir = model_dir
        self._configured_harness_name = harness_name
        self._configured_search_provider = search_provider
        self._configured_search_token_budget = search_token_budget

    def _clear_runtime(self) -> None:
        self._runtime_model = None
        self._tokenizer = None
        self._defaults = None
        self._engine = None
        self._harness = None
        self._runtime_search_token_budget = None

    def _require_runtime(self):
        if self._runtime_model is None or self._engine is None or self._harness is None:
            raise RuntimeError("LLM not started")
        return self._runtime_model, self._engine, self._harness

    def _require_running_runtime(self):
        runtime = self._require_runtime()
        if self._state == LLMState.RUNNING:
            return runtime
        if self._state in {LLMState.DRAINING, LLMState.SWAPPING}:
            raise AdmissionRejectedError(
                "LLM is draining and not accepting new requests"
            )
        raise RuntimeError("LLM is not running")

    async def _begin_swap(self) -> None:
        self._state = LLMState.DRAINING
        await self._admission.start_draining()
        for session in list(self._sessions):
            session._mark_stale()
        self._state = LLMState.SWAPPING
        await _wait_for_idle_or_cancel(self)

    async def _cancel_active_sessions(self) -> None:
        for session in list(self._sessions):
            await session.close()

    async def _recover_from_engine_failure(self) -> None:
        await restart_model(self)

    def _set_server_error(self) -> None:
        for session in list(self._sessions):
            session._mark_owner_stopped()
        self._clear_runtime()
        self._state = LLMState.SERVER_ERROR
