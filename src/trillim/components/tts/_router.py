"""HTTP router for the TTS component."""

from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from trillim.components.tts._limits import MAX_HTTP_TEXT_BYTES
from trillim.components.tts._validation import (
    PayloadTooLargeError,
    validate_http_speech_body,
    validate_http_speech_request,
    validate_http_voice_upload_request,
)
from trillim.errors import AdmissionRejectedError, InvalidRequestError, ProgressTimeoutError


def build_router(tts) -> APIRouter:
    """Build the TTS HTTP router."""
    router = APIRouter()

    @router.get("/v1/voices")
    async def list_voices():
        try:
            voices = await tts.list_voices()
        except Exception as exc:
            raise _as_http_error(exc) from exc
        return {"voices": voices}

    @router.post("/v1/voices")
    async def create_voice(request: Request):
        try:
            request.state.trillim_tts_voice_request = validate_http_voice_upload_request(
                content_length=request.headers.get("content-length"),
                name=request.headers.get("name"),
            )
            name = await tts._register_voice_http_request(request)
        except Exception as exc:
            raise _as_http_error(exc) from exc
        return {"name": name, "status": "created"}

    @router.delete("/v1/voices/{voice_name:path}")
    async def delete_voice(voice_name: str):
        try:
            deleted_name = await tts.delete_voice(voice_name)
        except Exception as exc:
            raise _as_http_error(exc) from exc
        return {"name": deleted_name, "status": "deleted"}

    @router.post("/v1/audio/speech")
    async def audio_speech(request: Request):
        try:
            speech_request = validate_http_speech_request(
                content_length=request.headers.get("content-length"),
                voice=request.headers.get("voice"),
                speed=request.headers.get("speed"),
                default_speed=tts.speed,
            )
            await tts._reject_if_busy()
            body = await _read_bounded_body(request, MAX_HTTP_TEXT_BYTES)
            text = validate_http_speech_body(body)
            session = await tts.speak(
                text,
                voice=speech_request.voice,
                speed=speech_request.speed,
            )
        except Exception as exc:
            raise _as_http_error(exc) from exc
        return StreamingResponse(
            _stream_speech_session(session),
            media_type="text/event-stream",
        )

    return router


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(f"speech input exceeds the {limit} byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _stream_speech_session(session):
    try:
        async with session:
            async for chunk in session:
                yield _sse("audio", base64.b64encode(chunk).decode("ascii"))
        yield _sse("done", "")
    except Exception as exc:
        yield _sse("error", str(exc).replace("\n", " "))


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
    return HTTPException(status_code=503, detail=str(exc))
