from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from trillim.components.stt._limits import (
    MAX_UPLOAD_BYTES,
    TOTAL_UPLOAD_TIMEOUT_SECONDS,
    UPLOAD_PROGRESS_TIMEOUT_SECONDS,
)
from trillim.components.stt._validation import PayloadTooLargeError, validate_http_request
from trillim.errors import ComponentLifecycleError, InvalidRequestError, ProgressTimeoutError


def build_router(stt) -> APIRouter:
    router = APIRouter()
    transcribe_lock = asyncio.Lock()

    @router.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request):
        if transcribe_lock.locked():
            raise HTTPException(status_code=429, detail="STT is already handling a request")
        await transcribe_lock.acquire()
        try:
            validated_request = validate_http_request(
                content_type=request.headers.get("content-type"),
                content_length=request.headers.get("content-length"),
                language=request.query_params.get("language"),
            )
            session = stt.open_session()
            audio_bytes = await _read_request_body(request)
            if not audio_bytes:
                raise InvalidRequestError("audio_bytes must not be empty")
            if validated_request.content_length is not None:
                actual_length = len(audio_bytes)
                if actual_length != validated_request.content_length:
                    raise InvalidRequestError("request body length did not match content-length")
            text = await session.transcribe(audio_bytes, language=validated_request.language)
        except Exception as exc:
            raise _as_http_error(exc) from exc
        finally:
            transcribe_lock.release()
        return {"text": text}

    return router


async def _read_request_body(request: Request) -> bytes:
    stream = request.stream().__aiter__()
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + UPLOAD_PROGRESS_TIMEOUT_SECONDS
    body = bytearray()
    while True:
        now = loop.time()
        remaining = min(deadline - now, started + TOTAL_UPLOAD_TIMEOUT_SECONDS - now)
        if remaining <= 0:
            raise ProgressTimeoutError("audio upload timed out")
        try:
            async with asyncio.timeout(remaining):
                chunk = await anext(stream)
        except StopAsyncIteration:
            return bytes(body)
        except TimeoutError as exc:
            raise ProgressTimeoutError("audio upload timed out") from exc
        if not chunk:
            continue
        body.extend(chunk)
        if len(body) > MAX_UPLOAD_BYTES:
            raise PayloadTooLargeError(f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit")
        deadline = loop.time() + UPLOAD_PROGRESS_TIMEOUT_SECONDS


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, PayloadTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidRequestError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ProgressTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    if isinstance(exc, ComponentLifecycleError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))
