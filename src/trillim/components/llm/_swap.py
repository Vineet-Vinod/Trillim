"""Hot-swap and recovery helpers for the LLM component."""

from __future__ import annotations

from trillim.components.llm._config import LLMState
from trillim.components.llm._limits import (
    SWAP_CANCELLATION_GRACE_SECONDS,
    SWAP_DRAIN_TIMEOUT_SECONDS,
)
from trillim.errors import AdmissionRejectedError, ComponentLifecycleError


class _StopWonError(ComponentLifecycleError):
    """Raised when stop preempts an in-flight model transition."""


async def swap_model(
    llm,
    init_config,
    *,
    harness_name: str | None = None,
    search_provider: str | None = None,
    search_token_budget: int | None = None,
) -> None:
    """Swap the active model after draining or cancelling in-flight work."""
    _mark_model_transition_active(llm)
    stop_epoch = _capture_stop_epoch(llm)
    try:
        if _stop_in_progress(llm):
            raise _StopWonError("LLM was stopped during model swap")
        _claim_model_transition(llm)
        try:
            async with llm._swap_lock:
                built_runtime = None
                old_engine = llm._engine
                try:
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        message="LLM was stopped during model swap",
                    )
                    built_runtime = llm._build_runtime(
                        init_config,
                        harness_name=harness_name,
                        search_provider=search_provider,
                        search_token_budget=search_token_budget,
                    )
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during model swap",
                    )
                    await llm._begin_swap()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during model swap",
                    )
                    if old_engine is not None:
                        await old_engine.stop()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during model swap",
                    )
                    llm._clear_runtime()
                    await built_runtime.engine.start()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during model swap",
                    )
                    llm._bind_runtime(built_runtime)
                    llm._state = LLMState.RUNNING
                    await llm._admission.finish_swapping()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during model swap",
                    )
                    llm._update_configured_runtime(
                        init_config=built_runtime.init_config,
                        harness_name=built_runtime.runtime_options.harness_name,
                        search_provider=built_runtime.runtime_options.search_provider,
                        search_token_budget=built_runtime.runtime_options.requested_search_token_budget,
                    )
                except _StopWonError:
                    raise
                except Exception:
                    if built_runtime is not None:
                        built_runtime.runtime_files.cleanup()
                        await _best_effort_stop(built_runtime.engine)
                    if built_runtime is None:
                        raise
                    await _best_effort_stop(old_engine)
                    llm._set_server_error()
                    raise
        finally:
            _release_model_transition(llm)
    finally:
        _finish_model_transition_active(llm)


async def restart_model(llm) -> None:
    """Restart the active model after a worker failure."""
    _mark_model_transition_active(llm)
    stop_epoch = _capture_stop_epoch(llm)
    try:
        if _stop_in_progress(llm):
            return
        try:
            _claim_model_transition(llm)
        except AdmissionRejectedError:
            if getattr(llm, "_state", None) == LLMState.RUNNING:
                set_server_error = getattr(llm, "_set_server_error", None)
                if set_server_error is not None:
                    set_server_error()
            return
        try:
            async with llm._swap_lock:
                if llm._runtime_model is None:
                    llm._set_server_error()
                    return
                old_engine = llm._engine
                built_runtime = None
                try:
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        message="LLM was stopped during recovery",
                    )
                    built_runtime = llm._build_runtime(
                        llm._configured_init_config,
                        harness_name=llm._configured_harness_name,
                        search_provider=llm._configured_search_provider,
                        search_token_budget=llm._configured_search_token_budget,
                    )
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during recovery",
                    )
                    await llm._begin_swap()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during recovery",
                    )
                    if old_engine is not None:
                        await old_engine.stop()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during recovery",
                    )
                    llm._clear_runtime()
                    await built_runtime.engine.start()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during recovery",
                    )
                    llm._bind_runtime(built_runtime)
                    llm._state = LLMState.RUNNING
                    await llm._admission.finish_swapping()
                    await _raise_if_stop_won(
                        llm,
                        stop_epoch,
                        built_runtime=built_runtime,
                        message="LLM was stopped during recovery",
                    )
                except _StopWonError:
                    return
                except Exception:
                    if built_runtime is not None:
                        built_runtime.runtime_files.cleanup()
                    await _best_effort_stop(
                        built_runtime.engine if built_runtime is not None else None
                    )
                    await _best_effort_stop(old_engine)
                    llm._set_server_error()
                    raise
        finally:
            _release_model_transition(llm)
    finally:
        _finish_model_transition_active(llm)


async def _wait_for_idle_or_cancel(llm) -> None:
    try:
        await llm._admission.wait_for_idle(timeout=SWAP_DRAIN_TIMEOUT_SECONDS)
        return
    except TimeoutError:
        await llm._cancel_active_sessions()
    try:
        await llm._admission.wait_for_idle(timeout=SWAP_CANCELLATION_GRACE_SECONDS)
        return
    except TimeoutError:
        if llm._engine is not None:
            await llm._engine.stop()
    try:
        await llm._admission.wait_for_idle(timeout=SWAP_CANCELLATION_GRACE_SECONDS)
    except TimeoutError as exc:
        raise RuntimeError("LLM failed to halt active generations during swap") from exc


async def _best_effort_stop(engine) -> None:
    if engine is None:
        return
    try:
        await engine.stop()
    except Exception:
        pass


def _capture_stop_epoch(llm):
    capture = getattr(llm, "_capture_stop_epoch", None)
    if capture is None:
        return None
    return capture()


def _stop_in_progress(llm) -> bool:
    checker = getattr(llm, "_stop_in_progress", None)
    if checker is None:
        return False
    return bool(checker())


def _claim_model_transition(llm) -> None:
    claim = getattr(llm, "_claim_model_transition", None)
    if claim is None:
        return
    claim()


def _release_model_transition(llm) -> None:
    release = getattr(llm, "_release_model_transition", None)
    if release is None:
        return
    release()


def _mark_model_transition_active(llm) -> None:
    mark = getattr(llm, "_mark_model_transition_active", None)
    if mark is None:
        return
    mark()


def _finish_model_transition_active(llm) -> None:
    finish = getattr(llm, "_finish_model_transition_active", None)
    if finish is None:
        return
    finish()


def _stop_requested_since(llm, stop_epoch) -> bool:
    if stop_epoch is None:
        return False
    checker = getattr(llm, "_stop_requested_since", None)
    if checker is None:
        return False
    return bool(checker(stop_epoch))


async def _discard_runtime_after_stop(llm, built_runtime) -> None:
    discard = getattr(llm, "_discard_runtime_after_stop", None)
    if discard is not None:
        await discard(built_runtime)
        return
    built_runtime.runtime_files.cleanup()
    await _best_effort_stop(getattr(built_runtime, "engine", None))


async def _raise_if_stop_won(
    llm,
    stop_epoch,
    *,
    built_runtime=None,
    message: str,
) -> None:
    if not _stop_requested_since(llm, stop_epoch):
        return
    if built_runtime is not None:
        await _discard_runtime_after_stop(llm, built_runtime)
    raise _StopWonError(message)
