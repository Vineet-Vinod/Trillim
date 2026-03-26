"""Public LLM component and chat session API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import weakref
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from fastapi import APIRouter

from trillim import _model_store
from trillim.components import Component
from trillim.components.llm._admission import GenerationAdmission
from trillim.components.llm._config import (
    InitConfig,
    LLMState,
    ModelInfo,
    RuntimeInitInfo,
    load_sampling_defaults,
)
from trillim.components.llm._engine import InferenceEngine, _first_protocol_line
from trillim.components.llm._limits import MAX_THREADS, TOKEN_PROGRESS_TIMEOUT_SECONDS
from trillim.components.llm._model_dir import (
    RuntimeFiles,
    prepare_runtime_files,
    validate_model_dir,
)
from trillim.components.llm._router import build_router
from trillim.components.llm._session import ChatSession, _ChatSession, _create_chat_session
from trillim.components.llm._swap import (
    _best_effort_stop,
    _wait_for_idle_or_cancel,
    restart_model,
    swap_model,
)
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
from trillim.errors import AdmissionRejectedError, ComponentLifecycleError, InvalidRequestError


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    harness_name: str
    search_provider: str
    search_token_budget: int
    requested_search_token_budget: int


@dataclass(frozen=True, slots=True)
class _BuiltRuntime:
    init_config: InitConfig
    runtime_files: RuntimeFiles
    validated: object
    tokenizer: object
    defaults: object
    engine: object
    runtime_options: _RuntimeOptions


class LLM(Component):
    """LLM component with truthful model state and bounded chat sessions."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        num_threads: int = 0,
        lora_dir: str | Path | None = None,
        lora_quant: str | None = None,
        unembed_quant: str | None = None,
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
        if search_token_budget < 1:
            raise ValueError("search_token_budget must be at least 1")
        self._configured_init_config = _make_init_config(
            model_dir=model_dir,
            num_threads=num_threads,
            lora_dir=lora_dir,
            lora_quant=lora_quant,
            unembed_quant=unembed_quant,
        )
        self._trust_remote_code = trust_remote_code
        self._configured_harness_name = validate_harness_name(harness_name)
        self._configured_search_provider = normalize_provider_name(search_provider)
        self._configured_search_token_budget = search_token_budget
        self._model_validator = _model_validator
        self._tokenizer_loader = _tokenizer_loader
        self._engine_factory = _engine_factory
        self._runtime_model = None
        self._runtime_init_config: InitConfig | None = None
        self._runtime_files: RuntimeFiles | None = None
        self._tokenizer = None
        self._defaults = None
        self._engine = None
        self._harness = None
        self._runtime_search_token_budget: int | None = None
        self._state = LLMState.UNAVAILABLE
        self._admission = GenerationAdmission()
        self._swap_lock = asyncio.Lock()
        self._stop_epoch = 0
        self._stop_requests = 0
        self._model_transition_claimed = False
        self._active_model_transitions = 0
        self._model_transitions_idle = asyncio.Event()
        self._model_transitions_idle.set()
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
        if self._stop_in_progress():
            raise ComponentLifecycleError("LLM is stopping")
        self._mark_model_transition_active()
        try:
            stop_epoch = self._capture_stop_epoch()
            built_runtime: _BuiltRuntime | None = None
            try:
                built_runtime = self._build_runtime(
                    self._configured_init_config,
                    harness_name=self._configured_harness_name,
                    search_provider=self._configured_search_provider,
                    search_token_budget=self._configured_search_token_budget,
                )
                if self._stop_requested_since(stop_epoch):
                    await self._discard_runtime_after_stop(built_runtime)
                    raise ComponentLifecycleError("LLM was stopped during startup")
                await built_runtime.engine.start()
                if self._stop_requested_since(stop_epoch):
                    await self._discard_runtime_after_stop(built_runtime)
                    raise ComponentLifecycleError("LLM was stopped during startup")
            except ComponentLifecycleError:
                raise
            except Exception:
                if built_runtime is not None:
                    built_runtime.runtime_files.cleanup()
                self._clear_runtime()
                self._state = LLMState.SERVER_ERROR
                raise
            self._bind_runtime(
                built_runtime,
            )
            self._update_configured_runtime(
                init_config=built_runtime.init_config,
                harness_name=built_runtime.runtime_options.harness_name,
                search_provider=built_runtime.runtime_options.search_provider,
                search_token_budget=built_runtime.runtime_options.requested_search_token_budget,
            )
            self._state = LLMState.RUNNING
            await self._admission.finish_swapping()
            if self._stop_requested_since(stop_epoch):
                await self._discard_runtime_after_stop(built_runtime)
                raise ComponentLifecycleError("LLM was stopped during startup")
        finally:
            self._finish_model_transition_active()

    async def stop(self) -> None:
        """Stop the inference engine and invalidate live sessions."""
        try:
            self._mark_stop_requested()
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
            await self._wait_for_model_transitions()
            self._state = LLMState.UNAVAILABLE
        finally:
            self._finish_stop_request()

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
        assert self._runtime_init_config is not None
        return ModelInfo(
            state=self._state,
            name=self._runtime_model.name,
            path=str(self._runtime_model.path),
            max_context_tokens=self._runtime_model.max_position_embeddings,
            trust_remote_code=self._trust_remote_code,
            adapter_path=(
                None
                if self._runtime_init_config.lora_dir is None
                else str(self._runtime_init_config.lora_dir)
            ),
            init_config=RuntimeInitInfo(
                num_threads=self._runtime_init_config.num_threads,
                lora_quant=self._runtime_init_config.lora_quant,
                unembed_quant=self._runtime_init_config.unembed_quant,
            ),
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
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator:
        """Stream one assistant turn from a temporary chat session."""
        async with self.open_session(messages) as session:
            async for event in session.stream_chat(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                rep_penalty_lookback=rep_penalty_lookback,
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
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Collect one assistant turn from a temporary chat session."""
        text, _usage = await self._collect_chat(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            rep_penalty_lookback=rep_penalty_lookback,
            max_tokens=max_tokens,
        )
        return text

    async def swap_model(
        self,
        model_dir: str | Path,
        *,
        num_threads: int | None = None,
        lora_dir: str | Path | None = None,
        lora_quant: str | None = None,
        unembed_quant: str | None = None,
        harness_name: str | None = None,
        search_provider: str | None = None,
        search_token_budget: int | None = None,
    ) -> ModelInfo:
        """Hot-swap to another model without restarting the server."""
        if self._state not in {LLMState.RUNNING, LLMState.SERVER_ERROR}:
            raise ComponentLifecycleError("LLM hot swap requires the component to be running")
        await swap_model(
            self,
            self._make_swap_init_config(
                model_dir,
                num_threads=num_threads,
                lora_dir=lora_dir,
                lora_quant=lora_quant,
                unembed_quant=unembed_quant,
            ),
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
        rep_penalty_lookback: int | None = None,
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
                rep_penalty_lookback=rep_penalty_lookback,
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
        init_config: InitConfig,
        *,
        harness_name: str | None = None,
        search_provider: str | None = None,
        search_token_budget: int | None = None,
    ):
        runtime_files = (
            prepare_runtime_files(
                init_config,
                trust_remote_code=self._trust_remote_code,
            )
            if self._model_validator is validate_model_dir or init_config.lora_dir is not None
            else RuntimeFiles(
                model_dir=Path(init_config.model_dir),
                metadata_dir=Path(init_config.model_dir),
            )
        )
        resolved_init_config = InitConfig(
            model_dir=runtime_files.model_dir,
            num_threads=init_config.num_threads,
            lora_dir=runtime_files.adapter_dir,
            lora_quant=init_config.lora_quant,
            unembed_quant=init_config.unembed_quant,
        )
        try:
            validated = self._validate_runtime_model(
                runtime_files.model_dir,
                metadata_dir=runtime_files.metadata_dir,
            )
            tokenizer = self._tokenizer_loader(
                runtime_files.metadata_dir,
                trust_remote_code=self._trust_remote_code,
            )
            defaults = load_sampling_defaults(runtime_files.metadata_dir)
            engine = self._engine_factory(
                validated,
                tokenizer,
                defaults,
                init_config=resolved_init_config,
                progress_timeout=TOKEN_PROGRESS_TIMEOUT_SECONDS,
            )
        except Exception:
            runtime_files.cleanup()
            raise
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
        return _BuiltRuntime(
            init_config=resolved_init_config,
            runtime_files=runtime_files,
            validated=validated,
            tokenizer=tokenizer,
            defaults=defaults,
            engine=engine,
            runtime_options=runtime_options,
        )

    def _bind_runtime(
        self,
        built_runtime: _BuiltRuntime,
    ) -> None:
        self._runtime_model = built_runtime.validated
        self._runtime_init_config = built_runtime.init_config
        self._runtime_files = built_runtime.runtime_files
        self._tokenizer = built_runtime.tokenizer
        self._defaults = built_runtime.defaults
        self._engine = built_runtime.engine
        self._runtime_search_token_budget = built_runtime.runtime_options.search_token_budget
        if built_runtime.runtime_options.harness_name == "default":
            self._harness = _DefaultHarness(built_runtime.engine)
            return
        self._harness = _SearchHarness(
            built_runtime.engine,
            search_provider=built_runtime.runtime_options.search_provider,
            search_token_budget=built_runtime.runtime_options.search_token_budget,
        )

    def _update_configured_runtime(
        self,
        *,
        init_config: InitConfig,
        harness_name: str,
        search_provider: str,
        search_token_budget: int,
    ) -> None:
        self._configured_init_config = init_config
        self._configured_harness_name = harness_name
        self._configured_search_provider = search_provider
        self._configured_search_token_budget = search_token_budget

    def _clear_runtime(self) -> None:
        runtime_files = self._runtime_files
        self._runtime_model = None
        self._runtime_init_config = None
        self._runtime_files = None
        self._tokenizer = None
        self._defaults = None
        self._engine = None
        self._harness = None
        self._runtime_search_token_budget = None
        if runtime_files is not None:
            runtime_files.cleanup()

    def _capture_stop_epoch(self) -> int:
        return self._stop_epoch

    def _stop_in_progress(self) -> bool:
        return self._stop_requests > 0

    def _stop_requested_since(self, epoch: int | None) -> bool:
        return epoch is not None and self._stop_epoch != epoch

    def _mark_stop_requested(self) -> None:
        self._stop_epoch += 1
        self._stop_requests += 1
        self._state = LLMState.UNAVAILABLE

    def _finish_stop_request(self) -> None:
        if self._stop_requests > 0:
            self._stop_requests -= 1

    def _claim_model_transition(self) -> None:
        if self._model_transition_claimed:
            raise AdmissionRejectedError("LLM hot swap is already in progress")
        self._model_transition_claimed = True

    def _release_model_transition(self) -> None:
        self._model_transition_claimed = False

    def _mark_model_transition_active(self) -> None:
        self._active_model_transitions += 1
        self._model_transitions_idle.clear()

    def _finish_model_transition_active(self) -> None:
        if self._active_model_transitions > 0:
            self._active_model_transitions -= 1
        if self._active_model_transitions == 0:
            self._model_transitions_idle.set()

    async def _wait_for_model_transitions(self) -> None:
        await self._model_transitions_idle.wait()

    async def _discard_runtime_after_stop(self, built_runtime: _BuiltRuntime) -> None:
        if (
            self._runtime_files is built_runtime.runtime_files
            or self._engine is built_runtime.engine
            or self._runtime_model is built_runtime.validated
        ):
            self._clear_runtime()
            await _best_effort_stop(built_runtime.engine)
            await self._admission.start_draining()
            self._state = LLMState.UNAVAILABLE
            return
        built_runtime.runtime_files.cleanup()
        await _best_effort_stop(built_runtime.engine)

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

    def _make_swap_init_config(
        self,
        model_dir: str | Path,
        *,
        num_threads: int | None,
        lora_dir: str | Path | None,
        lora_quant: str | None,
        unembed_quant: str | None,
    ) -> InitConfig:
        return _make_init_config(
            model_dir=model_dir,
            num_threads=0 if num_threads is None else num_threads,
            lora_dir=lora_dir,
            lora_quant=lora_quant,
            unembed_quant=unembed_quant,
        )

    def _validate_runtime_model(self, model_dir: Path, *, metadata_dir: Path):
        if self._model_validator is validate_model_dir:
            return self._model_validator(model_dir, metadata_dir=metadata_dir)
        return self._model_validator(model_dir)


def _make_init_config(
    *,
    model_dir: str | Path,
    num_threads: int,
    lora_dir: str | Path | None,
    lora_quant: str | None,
    unembed_quant: str | None,
) -> InitConfig:
    if not model_dir:
        raise ValueError("model_dir is required")
    if num_threads < 0 or num_threads > MAX_THREADS:
        raise ValueError(f"num_threads must be between 0 and {MAX_THREADS}")
    return InitConfig(
        model_dir=_normalize_store_id("model_dir", model_dir),
        num_threads=num_threads,
        lora_dir=_normalize_optional_store_id("lora_dir", lora_dir),
        lora_quant=_normalize_optional_text("lora_quant", lora_quant),
        unembed_quant=_normalize_optional_text("unembed_quant", unembed_quant),
    )


def _normalize_store_id(field_name: str, value: str | Path) -> Path:
    normalized = _first_protocol_line(str(value))
    if not normalized.strip():
        raise ValueError(f"{field_name} must not be empty")
    try:
        return _model_store.resolve_existing_store_id(
            normalized,
            error_type=InvalidRequestError,
        )
    except InvalidRequestError as exc:
        raise InvalidRequestError(f"{field_name}: {exc}") from exc


def _normalize_optional_store_id(field_name: str, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return _normalize_store_id(field_name, value)


def _normalize_optional_text(field_name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _first_protocol_line(value)
    if not normalized.strip():
        raise ValueError(f"{field_name} must not be empty")
    return normalized
