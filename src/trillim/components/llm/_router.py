"""HTTP router for the LLM component."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from trillim.components.llm._config import ModelInfo
from trillim.components.llm._events import ChatDoneEvent, ChatTokenEvent
from trillim.components.llm._limits import REQUEST_BODY_LIMIT_BYTES
from trillim.components.llm._validation import validate_chat_request, validate_swap_request
from trillim.errors import (
    AdmissionRejectedError,
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

    @router.get("/v1/models")
    async def list_models():
        info = llm.model_info()
        return {
            "object": "list",
            "state": info.state,
            "data": [] if info.name is None else [_model_payload(info)],
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        info = llm.model_info()
        payload = await _read_json_body(request, REQUEST_BODY_LIMIT_BYTES)
        try:
            chat_request = validate_chat_request(
                payload,
                active_model_name=info.name,
            )
            if chat_request.stream:
                return StreamingResponse(
                    _stream_chat_response(llm, chat_request),
                    media_type="text/event-stream",
                )
            text, usage = await llm._collect_chat(
                [
                    {"role": message.role, "content": message.content}
                    for message in chat_request.messages
                ],
                temperature=chat_request.temperature,
                top_k=chat_request.top_k,
                top_p=chat_request.top_p,
                repetition_penalty=chat_request.repetition_penalty,
                max_tokens=chat_request.max_tokens,
            )
        except Exception as exc:
            raise _as_http_error(exc) from exc
        model_info = llm.model_info()
        response_id = _response_id()
        created = int(time.time())
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_info.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cached_tokens": usage.cached_tokens,
            },
        }

    if allow_hot_swap:

        @router.post("/v1/models/swap")
        async def swap_model_route(request: Request):
            payload = await _read_json_body(request, REQUEST_BODY_LIMIT_BYTES)
            try:
                swap_request = validate_swap_request(payload)
                info = await llm.swap_model(swap_request.model_dir)
            except Exception as exc:
                raise _as_http_error(exc) from exc
            return {
                "object": "model.swap",
                "model": info.name,
                "state": info.state,
                "path": info.path,
            }

    return router


async def _stream_chat_response(llm, request_model):
    response_id = _response_id()
    created = int(time.time())
    yield _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": llm.model_info().name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )
    async with llm.open_session(
        [
            {"role": message.role, "content": message.content}
            for message in request_model.messages
        ]
    ) as session:
        async for event in session.stream_chat(
            temperature=request_model.temperature,
            top_k=request_model.top_k,
            top_p=request_model.top_p,
            repetition_penalty=request_model.repetition_penalty,
            max_tokens=request_model.max_tokens,
        ):
            if isinstance(event, ChatTokenEvent):
                if not event.text:
                    continue
                yield _sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": llm.model_info().name,
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
                        "model": llm.model_info().name,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    }
                )
                break
    yield "data: [DONE]\n\n"


def _model_payload(info: ModelInfo) -> dict[str, object]:
    return {
        "id": info.name,
        "object": "model",
        "path": info.path,
        "max_context_tokens": info.max_context_tokens,
        "trust_remote_code": info.trust_remote_code,
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
    if isinstance(exc, (InvalidRequestError, ModelValidationError, ContextOverflowError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (SessionClosedError, SessionExhaustedError, SessionStaleError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, AdmissionRejectedError):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ProgressTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))


def _response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
