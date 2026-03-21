"""HTTP router for the STT component."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from trillim.components.stt._validation import PayloadTooLargeError
from trillim.errors import AdmissionRejectedError, InvalidRequestError, ProgressTimeoutError


def build_router(stt) -> APIRouter:
    """Build the Phase 4 raw-body STT router."""
    router = APIRouter()

    @router.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request):
        try:
            text = await _handle_transcription_request(stt, request)
        except Exception as exc:
            raise _as_http_error(exc) from exc
        return {"text": text}

    return router


async def _handle_transcription_request(stt, request: Request) -> str:
    return await stt._transcribe_http_request(request)


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, PayloadTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidRequestError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, AdmissionRejectedError):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ProgressTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))
