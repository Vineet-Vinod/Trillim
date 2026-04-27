"""Public LLM component API."""

from __future__ import annotations

import asyncio
import threading
from asyncio import AbstractEventLoop
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter

from trillim import _model_store
from trillim.components import Component
from trillim.components.llm._config import (
    _HarnessConfig,
    InitConfig,
    ModelRuntimeConfig,
    SamplingDefaults,
    load_sampling_defaults,
)
from trillim.components.llm._engine import (
    EngineCrashedError,
    EngineError,
    EngineProgressTimeoutError,
    InferenceEngine,
    _first_protocol_line,
)
from trillim.components.llm._limits import MAX_THREADS, TOKEN_PROGRESS_TIMEOUT_SECONDS
from trillim.components.llm._model_dir import (
    RuntimeFiles,
    prepare_runtime_files,
    validate_model_dir,
)
from trillim.components.llm._router import build_router
from trillim.components.llm._session import ChatSession, _create_chat_session
from trillim.harnesses._default import _DefaultHarness
from trillim.harnesses.search._harness import _SearchHarness
from trillim.harnesses.search.provider import (
    DEFAULT_SEARCH_TOKEN_BUDGET,
    normalize_provider_name,
    resolve_search_token_budget,
    validate_harness_name,
)
from trillim.errors import (
    ComponentLifecycleError,
    InvalidRequestError,
    ModelValidationError,
    SessionStaleError,
)


@dataclass(slots=True)
class _RuntimeSnapshot:
    model: ModelRuntimeConfig
    init_config: InitConfig
    runtime_files: RuntimeFiles
    tokenizer: object
    defaults: SamplingDefaults
    engine: InferenceEngine
    harness_config: _HarnessConfig


def load_tokenizer(model_dir: Path, *, trust_remote_code: bool):
    """Load a tokenizer from a validated model directory."""
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ModelValidationError("transformers is required to load tokenizers") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise ModelValidationError(
            f"Could not load tokenizer from {model_dir}"
        ) from exc
    if not callable(getattr(tokenizer, "encode", None)) or not callable(
        getattr(tokenizer, "decode", None)
    ):
        raise ModelValidationError(
            f"Tokenizer loaded from {model_dir} is missing encode/decode methods"
        )
    return tokenizer


class LLM(Component):
    """LLM component owning one long-lived runtime and one generation lane."""

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
        allow_hot_swap: bool = False,
        _model_validator=validate_model_dir,
        _tokenizer_loader=load_tokenizer,
        _engine_factory=InferenceEngine,
    ) -> None:
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
        self._allow_hot_swap = allow_hot_swap
        self._model_validator = _model_validator
        self._tokenizer_loader = _tokenizer_loader
        self._engine_factory = _engine_factory

        self._runtime: _RuntimeSnapshot | None = None
        self._runtime_epoch = 0
        self._runtime_epoch_lock = threading.Lock()
        self._generation_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._loop: AbstractEventLoop | None = None
        self._started = False

    def router(self) -> APIRouter:
        """Return the FastAPI router for this LLM component."""
        return build_router(self, allow_hot_swap=self._allow_hot_swap)

    async def start(self) -> None:
        """Load the configured model and start the inference engine."""
        self._require_owner_loop()
        async with self._lifecycle_lock:
            if self._runtime is not None and self._started:
                return
            built_runtime: _RuntimeSnapshot | None = None
            try:
                built_runtime = self._build_runtime(
                    self._configured_init_config,
                    harness_name=self._configured_harness_name,
                    search_provider=self._configured_search_provider,
                    search_token_budget=self._configured_search_token_budget,
                )
                await built_runtime.engine.start()
            except Exception:
                if built_runtime is not None:
                    built_runtime.runtime_files.cleanup()
                self._clear_runtime(increment_epoch=True)
                self._started = False
                raise
            self._bind_runtime(built_runtime)
            self._started = True

    async def stop(self) -> None:
        """Stop the inference engine and clear in-memory runtime state."""
        self._require_owner_loop()
        async with self._lifecycle_lock:
            runtime = self._runtime
            if runtime is None:
                self._started = False
                return
            stop_error: Exception | None = None
            try:
                async with self._generation_lock:
                    self._runtime = None
                    self._increment_runtime_epoch()
                    try:
                        await runtime.engine.stop()
                    except Exception as exc:
                        stop_error = exc
                    finally:
                        try:
                            runtime.runtime_files.cleanup()
                        except Exception as exc:
                            if stop_error is None:
                                stop_error = exc
                    if stop_error is not None:
                        raise stop_error
            except Exception:
                self._started = False
                raise
            self._started = False

    def open_session(self) -> ChatSession:
        """Create a new chat session bound to the current runtime snapshot."""
        self._require_owner_loop()
        runtime = self._require_running_runtime()
        return _create_chat_session(self, runtime, self._current_runtime_epoch())

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
    ) -> None:
        """Hot-swap to another model runtime."""
        self._require_owner_loop()
        async with self._lifecycle_lock:
            if not self._started or self._runtime is None:
                raise ComponentLifecycleError(
                    "LLM hot swap requires the component to be running"
                )
            async with self._generation_lock:
                old_runtime = self._runtime
                new_runtime: _RuntimeSnapshot | None = None
                old_teardown_started = False
                try:
                    new_runtime = self._build_runtime(
                        _make_init_config(
                            model_dir=model_dir,
                            num_threads=(
                                self._configured_init_config.num_threads
                                if num_threads is None
                                else num_threads
                            ),
                            lora_dir=lora_dir,
                            lora_quant=lora_quant,
                            unembed_quant=unembed_quant,
                        ),
                        harness_name=harness_name,
                        search_provider=search_provider,
                        search_token_budget=search_token_budget,
                    )
                    if old_runtime is not None:
                        old_teardown_started = True
                        teardown_error: Exception | None = None
                        try:
                            await old_runtime.engine.stop()
                        except Exception as exc:
                            teardown_error = exc
                        finally:
                            try:
                                old_runtime.runtime_files.cleanup()
                            except Exception as exc:
                                if teardown_error is None:
                                    teardown_error = exc
                        if teardown_error is not None:
                            raise teardown_error
                    await new_runtime.engine.start()
                except Exception:
                    if new_runtime is not None:
                        try:
                            new_runtime.runtime_files.cleanup()
                        except Exception:
                            pass
                    if old_teardown_started:
                        self._runtime = None
                        self._increment_runtime_epoch()
                        self._started = False
                    raise
                self._runtime = new_runtime
                self._configured_init_config = new_runtime.init_config
                self._configured_harness_name = new_runtime.harness_config.name
                self._configured_search_provider = new_runtime.harness_config.search_provider
                self._configured_search_token_budget = (
                    new_runtime.harness_config.requested_search_token_budget
                )
                self._started = True
                self._increment_runtime_epoch()

    def _active_model_name(self) -> str | None:
        runtime = self._runtime
        if not self._started or runtime is None:
            return None
        return runtime.model.name

    async def _generate(
        self,
        runtime: _RuntimeSnapshot,
        runtime_epoch: int,
        *,
        token_ids: Sequence[int],
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        rep_penalty_lookback: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[int]:
        self._require_owner_loop()
        self._require_runtime_epoch(runtime_epoch)
        reached_engine = False
        async with self._generation_lock:
            self._require_running_runtime()
            self._require_runtime_epoch(runtime_epoch)
            try:
                reached_engine = True
                async for token_id in runtime.engine.generate(
                    token_ids=token_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    rep_penalty_lookback=rep_penalty_lookback,
                    max_tokens=max_tokens,
                ):
                    yield token_id
            except (asyncio.CancelledError, GeneratorExit):
                if reached_engine:
                    await self._recover_engine(runtime)
                raise
            except (EngineProgressTimeoutError, EngineCrashedError, EngineError):
                await self._recover_engine(runtime)
                raise

    def _build_harness(self, runtime: _RuntimeSnapshot):
        config = runtime.harness_config
        if config.name == "default":
            return _DefaultHarness(self, runtime)
        return _SearchHarness(
            self,
            runtime,
            search_provider=config.search_provider,
            search_token_budget=config.search_token_budget,
        )

    async def _recover_engine(self, runtime: _RuntimeSnapshot) -> None:
        if self._runtime is not runtime:
            return
        try:
            await runtime.engine.recover()
            self._started = True
        except Exception:
            try:
                await runtime.engine.stop()
            except Exception:
                pass
            self._clear_runtime(increment_epoch=True)
            self._started = False
            raise

    def _build_runtime(
        self,
        init_config: InitConfig,
        *,
        harness_name: str | None = None,
        search_provider: str | None = None,
        search_token_budget: int | None = None,
    ) -> _RuntimeSnapshot:
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
            model = self._validate_runtime_model(
                runtime_files.model_dir,
                metadata_dir=runtime_files.metadata_dir,
            )
            tokenizer = self._tokenizer_loader(
                runtime_files.metadata_dir,
                trust_remote_code=self._trust_remote_code,
            )
            defaults = load_sampling_defaults(runtime_files.metadata_dir)
            engine = self._engine_factory(
                model,
                tokenizer,
                defaults,
                init_config=resolved_init_config,
                progress_timeout=TOKEN_PROGRESS_TIMEOUT_SECONDS,
            )
            requested_budget = (
                self._configured_search_token_budget
                if search_token_budget is None
                else search_token_budget
            )
            harness_config = _HarnessConfig(
                name=validate_harness_name(
                    self._configured_harness_name if harness_name is None else harness_name
                ),
                search_provider=normalize_provider_name(
                    self._configured_search_provider
                    if search_provider is None
                    else search_provider
                ),
                search_token_budget=resolve_search_token_budget(
                    requested_budget,
                    max_context_tokens=model.max_position_embeddings,
                ),
                requested_search_token_budget=requested_budget,
            )
        except Exception:
            runtime_files.cleanup()
            raise
        return _RuntimeSnapshot(
            model=model,
            init_config=resolved_init_config,
            runtime_files=runtime_files,
            tokenizer=tokenizer,
            defaults=defaults,
            engine=engine,
            harness_config=harness_config,
        )

    def _bind_runtime(self, runtime: _RuntimeSnapshot) -> None:
        old_runtime = self._runtime
        self._runtime = runtime
        self._configured_init_config = runtime.init_config
        self._configured_harness_name = runtime.harness_config.name
        self._configured_search_provider = runtime.harness_config.search_provider
        self._configured_search_token_budget = (
            runtime.harness_config.requested_search_token_budget
        )
        if old_runtime is not runtime:
            self._increment_runtime_epoch()

    def _clear_runtime(self, *, increment_epoch: bool) -> None:
        runtime = self._runtime
        self._runtime = None
        if increment_epoch:
            self._increment_runtime_epoch()
        if runtime is not None:
            try:
                runtime.runtime_files.cleanup()
            except Exception:
                pass

    def _require_running_runtime(self) -> _RuntimeSnapshot:
        runtime = self._runtime
        if runtime is None or not self._started:
            raise ComponentLifecycleError("LLM is not running")
        return runtime

    def _require_runtime_epoch(self, runtime_epoch: int) -> None:
        if runtime_epoch != self._current_runtime_epoch():
            raise SessionStaleError(
                "ChatSession is stale after the active model changed; create a new session"
            )

    def _current_runtime_epoch(self) -> int:
        with self._runtime_epoch_lock:
            return self._runtime_epoch

    def _increment_runtime_epoch(self) -> None:
        with self._runtime_epoch_lock:
            self._runtime_epoch += 1

    def _require_owner_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            return
        if loop is not self._loop:
            raise ComponentLifecycleError(
                "LLM is bound to one event loop; create a new LLM per thread/event loop"
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
