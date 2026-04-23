from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from pathlib import Path

from trillim.components import Component
from trillim.components.stt._admission import TranscriptionAdmission
from trillim.components.stt._engine import STTEngine
from trillim.components.stt._limits import (
    MAX_UPLOAD_BYTES,
    TOTAL_UPLOAD_TIMEOUT_SECONDS,
    UPLOAD_PROGRESS_TIMEOUT_SECONDS,
)
from trillim.components.stt._router import build_router
from trillim.components.stt._session import AudioSession, _create_audio_session
from trillim.components.stt._validation import PayloadTooLargeError, validate_http_request
from trillim.errors import ComponentLifecycleError, InvalidRequestError, ProgressTimeoutError


class STT(Component):
    def __init__(self) -> None:
        self._engine = STTEngine()
        self._transcribe_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._router_admission = TranscriptionAdmission()
        self._stop_event = asyncio.Event()
        self._owner_loop: AbstractEventLoop | None = None
        self._started = False

    def router(self):
        return build_router(self)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            self._require_owner_loop()
            if self._started:
                return
            await self._engine.start()
            await self._router_admission.finish_starting()
            self._stop_event.clear()
            self._started = True

    async def stop(self) -> None:
        self._require_owner_loop()
        async with self._lifecycle_lock:
            if not self._started and self._stop_event.is_set():
                return
            await self._router_admission.start_draining()
            self._stop_event.set()
            self._started = False
            async with self._transcribe_lock:
                await self._engine.stop()
        await self._router_admission.wait_for_idle()

    def open_session(self) -> AudioSession:
        self._require_owner_loop()
        if not self._started or self._stop_event.is_set():
            raise ComponentLifecycleError("STT is not running")
        return _create_audio_session(self)

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
    ) -> str:
        return await self.open_session().transcribe(audio_bytes, language=language)

    async def transcribe_file(
        self,
        path: str | Path,
        *,
        language: str | None = None,
    ) -> str:
        source_path = self._validate_source_file(path)
        return await self.open_session().transcribe(source_path.read_bytes(), language=language)

    async def _transcribe_http_request(self, request) -> str:
        self._require_owner_loop()
        self._require_started()
        validated_request = validate_http_request(
            content_type=request.headers.get("content-type"),
            content_length=request.headers.get("content-length"),
            language=request.query_params.get("language"),
        )
        async with await self._router_admission.acquire():
            audio_bytes = await self._read_request_body(request)
            if not audio_bytes:
                raise InvalidRequestError("audio_bytes must not be empty")
            if validated_request.content_length is not None:
                actual_length = len(audio_bytes)
                if actual_length != validated_request.content_length:
                    raise InvalidRequestError("request body length did not match content-length")
            return await self.transcribe_bytes(audio_bytes, language=validated_request.language)

    async def _transcribe(self, pcm: bytes, *, language: str | None = None) -> str:
        self._require_owner_loop()
        if not self._started or self._stop_event.is_set():
            return ""
        async with self._transcribe_lock:
            if not self._started or self._stop_event.is_set():
                return ""
            return await self._engine.transcribe(pcm, language=language)

    def _require_owner_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
            return
        if loop is not self._owner_loop:
            raise ComponentLifecycleError(
                "STT is bound to one event loop; create a new STT per thread/event loop"
            )

    def _validate_source_file(self, path: str | Path) -> Path:
        if isinstance(path, str) and not path:
            raise InvalidRequestError("path is required")
        return Path(path).expanduser()

    def _require_started(self) -> None:
        if not self._started or self._stop_event.is_set():
            raise ComponentLifecycleError("STT is not running")

    async def _read_request_body(self, request) -> bytes:
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
                raise PayloadTooLargeError(
                    f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit"
                )
            deadline = loop.time() + UPLOAD_PROGRESS_TIMEOUT_SECONDS
