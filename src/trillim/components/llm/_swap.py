"""Hot-swap and recovery helpers for the LLM component."""

from __future__ import annotations

from trillim.components.llm._config import LLMState
from trillim.components.llm._limits import (
    SWAP_CANCELLATION_GRACE_SECONDS,
    SWAP_DRAIN_TIMEOUT_SECONDS,
)


async def swap_model(llm, model_dir: str) -> None:
    """Swap the active model after draining or cancelling in-flight work."""
    async with llm._swap_lock:
        validated, tokenizer, defaults, engine = llm._build_runtime(model_dir)
        old_engine = llm._engine
        try:
            await llm._begin_swap()
            if old_engine is not None:
                await old_engine.stop()
            llm._clear_runtime()
            await engine.start()
            llm._bind_runtime(validated, tokenizer, defaults, engine)
            llm._state = LLMState.RUNNING
            await llm._admission.finish_swapping()
        except Exception:
            await _best_effort_stop(engine)
            await _best_effort_stop(old_engine)
            llm._set_server_error()
            raise


async def restart_model(llm) -> None:
    """Restart the active model after a worker failure."""
    async with llm._swap_lock:
        if llm._runtime_model is None:
            llm._set_server_error()
            return
        old_engine = llm._engine
        try:
            validated, tokenizer, defaults, engine = llm._build_runtime(
                llm._runtime_model.path
            )
            await llm._begin_swap()
            if old_engine is not None:
                await old_engine.stop()
            llm._clear_runtime()
            await engine.start()
            llm._bind_runtime(validated, tokenizer, defaults, engine)
            llm._state = LLMState.RUNNING
            await llm._admission.finish_swapping()
        except Exception:
            await _best_effort_stop(engine if "engine" in locals() else None)
            await _best_effort_stop(old_engine)
            llm._set_server_error()
            raise


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
