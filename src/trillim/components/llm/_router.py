"""HTTP router for the LLM component."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from trillim.components.llm._events import ChatDoneEvent, ChatTokenEvent
from trillim.components.llm._limits import REQUEST_BODY_LIMIT_BYTES
from trillim.components.llm._validation import validate_chat_request, validate_swap_request
from trillim.errors import (
    AdmissionRejectedError,
    ComponentLifecycleError,
    ContextOverflowError,
    InvalidRequestError,
    ModelValidationError,
    ProgressTimeoutError,
    SessionClosedError,
    SessionExhaustedError,
    SessionStaleError,
)


def build_router(llm, *, allow_hot_swap: bool) -> APIRouter:
    """Build the HTTP router for an LLM component instance."""
    router = APIRouter()
    request_lock = asyncio.Lock()

    @router.get("/v1/models")
    async def list_models():
        if request_lock.locked():
            raise _as_http_error(AdmissionRejectedError("LLM is already handling a request"))
        await request_lock.acquire()
        try:
            model_name = llm._active_model_name()
        except Exception as exc:
            raise _as_http_error(exc) from exc
        finally:
            request_lock.release()
        return {
            "object": "list",
            "data": [] if model_name is None else [_model_payload(model_name)],
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await _read_json_body(request, REQUEST_BODY_LIMIT_BYTES)
        if request_lock.locked():
            raise _as_http_error(AdmissionRejectedError("LLM is already handling a request"))
        await request_lock.acquire()
        try:
            model_name = llm._active_model_name()
            if model_name is None:
                raise ComponentLifecycleError("LLM is not running")
            chat_request = validate_chat_request(payload, active_model_name=model_name)
            response_id = _response_id()
            created = int(time.time())
            if chat_request.stream:
                session = None
                try:
                    session = _session_for_request(llm, chat_request)
                    return StreamingResponse(
                        _stream_chat_response(
                            session,
                            final_user_content=chat_request.messages[-1].content,
                            sampling=_sampling_kwargs(chat_request),
                            response_id=response_id,
                            created=created,
                            model_name=model_name,
                            request_lock=request_lock,
                        ),
                        media_type="text/event-stream",
                    )
                except Exception:
                    if session is not None:
                        await session.close()
                    request_lock.release()
                    raise
            session = None
            try:
                session = _session_for_request(llm, chat_request)
                text = await session.collect(
                    chat_request.messages[-1].content,
                    **_sampling_kwargs(chat_request),
                )
                usage_payload = _usage_payload(getattr(session, "_last_usage", None))
            finally:
                if session is not None:
                    await session.close()
                request_lock.release()
        except Exception as exc:
            if request_lock.locked():
                request_lock.release()
            raise _as_http_error(exc) from exc
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_payload,
        }

    if allow_hot_swap:

        @router.post("/v1/models/swap")
        async def swap_model_route(request: Request):
            payload = await _read_json_body(request, REQUEST_BODY_LIMIT_BYTES)
            try:
                swap_request = validate_swap_request(payload)
                if request_lock.locked():
                    raise AdmissionRejectedError("LLM is already handling a request")
                await request_lock.acquire()
                try:
                    await llm.swap_model(
                        swap_request.model_dir,
                        num_threads=swap_request.num_threads,
                        lora_dir=swap_request.lora_dir,
                        lora_quant=swap_request.lora_quant,
                        unembed_quant=swap_request.unembed_quant,
                        harness_name=swap_request.harness_name,
                        search_provider=swap_request.search_provider,
                        search_token_budget=swap_request.search_token_budget,
                    )
                    model_name = llm._active_model_name()
                finally:
                    request_lock.release()
            except Exception as exc:
                raise _as_http_error(exc) from exc
            return {
                "object": "model.swap",
                "model": model_name,
            }

    return router


def _session_for_request(llm, request_model):
    session = llm.open_session()
    for message in request_model.messages[:-1]:
        session.append_message(message.role, message.content)
    return session


async def _stream_chat_response(
    session,
    *,
    final_user_content: str,
    sampling: dict[str, float | int | None],
    response_id: str,
    created: int,
    model_name: str | None,
    request_lock: asyncio.Lock,
) -> AsyncIterator[str]:
    try:
        yield _sse(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
        )
        async for event in session.generate(final_user_content, **sampling):
            if isinstance(event, ChatTokenEvent):
                if not event.text:
                    continue
                yield _sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": event.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            elif isinstance(event, ChatDoneEvent):
                yield _sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    }
                )
                break
        yield "data: [DONE]\n\n"
    except Exception as exc:
        yield _sse({"error": {"message": str(exc), "type": exc.__class__.__name__}})
    finally:
        await session.close()
        request_lock.release()


def _model_payload(model_name: str) -> dict[str, object]:
    return {
        "id": model_name,
        "object": "model",
    }


async def _read_json_body(request: Request, limit: int) -> object:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length header") from exc
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(
        exc,
        (
            InvalidRequestError,
            ModelValidationError,
            ContextOverflowError,
        ),
    ):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ComponentLifecycleError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, (SessionClosedError, SessionExhaustedError, SessionStaleError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, AdmissionRejectedError):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ProgressTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))


def _sampling_kwargs(request_model) -> dict[str, float | int | None]:
    return {
        "temperature": request_model.temperature,
        "top_k": request_model.top_k,
        "top_p": request_model.top_p,
        "repetition_penalty": request_model.repetition_penalty,
        "rep_penalty_lookback": request_model.rep_penalty_lookback,
        "max_tokens": request_model.max_tokens,
    }


def _usage_payload(usage) -> dict[str, int]:
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cached_tokens": usage.cached_tokens,
    }


def _response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
