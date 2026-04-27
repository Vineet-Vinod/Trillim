"""HTTP router for the TTS component."""

from __future__ import annotations

import asyncio
import base64

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from trillim.components.tts._limits import (
    DEFAULT_SPEED,
    MAX_HTTP_TEXT_BYTES,
    MAX_VOICE_UPLOAD_BYTES,
    PROGRESS_TIMEOUT_SECONDS,
    TOTAL_UPLOAD_TIMEOUT_SECONDS,
)
from trillim.components.tts._validation import (
    PayloadTooLargeError,
    validate_http_speech_body,
    validate_http_speech_request,
    validate_http_voice_upload_request,
)
from trillim.errors import (
    AdmissionRejectedError,
    ComponentLifecycleError,
    InvalidRequestError,
    ProgressTimeoutError,
)


def build_router(tts) -> APIRouter:
    """Build the TTS HTTP router."""
    router = APIRouter()
    request_lock = asyncio.Lock()

    @router.get("/v1/voices")
    async def list_voices():
        if request_lock.locked():
            raise HTTPException(status_code=429, detail="TTS is already handling a request")
        await request_lock.acquire()
        try:
            voices = await tts.list_voices()
        except Exception as exc:
            raise _as_http_error(exc) from exc
        finally:
            request_lock.release()
        return {"voices": voices}

    @router.post("/v1/voices")
    async def create_voice(request: Request):
        if request_lock.locked():
            raise HTTPException(status_code=429, detail="TTS is already handling a request")
        await request_lock.acquire()
        try:
            metadata = validate_http_voice_upload_request(
                content_length=request.headers.get("content-length"),
                name=request.headers.get("name"),
            )
            body = await _read_bounded_body(request, MAX_VOICE_UPLOAD_BYTES)
            _validate_actual_content_length(body, metadata.content_length)
            name = await tts.register_voice(metadata.name, body)
        except Exception as exc:
            raise _as_http_error(exc) from exc
        finally:
            request_lock.release()
        return {"name": name, "status": "created"}

    @router.delete("/v1/voices/{voice_name}")
    async def delete_voice(voice_name: str):
        if request_lock.locked():
            raise HTTPException(status_code=429, detail="TTS is already handling a request")
        await request_lock.acquire()
        try:
            deleted_name = await tts.delete_voice(voice_name)
        except Exception as exc:
            raise _as_http_error(exc) from exc
        finally:
            request_lock.release()
        return {"name": deleted_name, "status": "deleted"}

    @router.post("/v1/audio/speech")
    async def audio_speech(request: Request):
        try:
            speech_request = validate_http_speech_request(
                content_length=request.headers.get("content-length"),
                voice=request.headers.get("voice"),
                speed=request.headers.get("speed"),
                default_speed=DEFAULT_SPEED,
            )
        except Exception as exc:
            raise _as_http_error(exc) from exc

        if request_lock.locked():
            raise _as_http_error(AdmissionRejectedError("TTS is busy; only one live session is allowed"))
        await request_lock.acquire()

        try:
            body = await _read_bounded_body(request, MAX_HTTP_TEXT_BYTES)
            _validate_actual_content_length(body, speech_request.content_length)
            text = validate_http_speech_body(body)
            session = await tts.open_session(
                voice=speech_request.voice,
                speed=speech_request.speed,
            )
        except Exception as exc:
            request_lock.release()
            raise _as_http_error(exc) from exc
        return StreamingResponse(
            _stream_speech_session(session, text, request_lock),
            media_type="text/event-stream",
        )

    return router


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    total = 0
    chunks: list[bytes] = []
    stream = request.stream().__aiter__()
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + PROGRESS_TIMEOUT_SECONDS
    while True:
        now = loop.time()
        remaining = min(deadline - now, started + TOTAL_UPLOAD_TIMEOUT_SECONDS - now)
        if remaining <= 0:
            raise ProgressTimeoutError("speech input upload timed out")
        try:
            async with asyncio.timeout(remaining):
                chunk = await anext(stream)
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            raise ProgressTimeoutError("speech input upload timed out") from exc
        if not chunk:
            continue
        deadline = loop.time() + PROGRESS_TIMEOUT_SECONDS
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(f"speech input exceeds the {limit} byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_actual_content_length(body: bytes, content_length: int | None) -> None:
    if content_length is not None and len(body) != content_length:
        raise InvalidRequestError("request body length did not match content-length")


async def _stream_speech_session(session, text: str, request_lock: asyncio.Lock):
    try:
        async for chunk in session.synthesize(text):
            yield _sse("audio", base64.b64encode(chunk).decode("ascii"))
        yield _sse("done", "")
    except Exception as exc:
        yield _sse("error", str(exc).replace("\n", " "))
    finally:
        await session.close()
        request_lock.release()


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc.args[0]))
    if isinstance(exc, PayloadTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidRequestError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, AdmissionRejectedError):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ProgressTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    if isinstance(exc, ComponentLifecycleError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))
