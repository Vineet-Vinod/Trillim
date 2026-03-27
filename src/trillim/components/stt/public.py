"""Public Phase 4 STT component."""

from __future__ import annotations

import asyncio
import importlib
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Request

from trillim.components import Component
from trillim.components.stt._admission import TranscriptionAdmission
from trillim.components.stt._limits import (
    TOTAL_UPLOAD_TIMEOUT_SECONDS,
    UPLOAD_PROGRESS_TIMEOUT_SECONDS,
)
from trillim.components.stt._router import build_router
from trillim.components.stt._spool import copy_source_file, spool_audio_bytes, spool_request_stream
from trillim.components.stt._validation import (
    validate_audio_bytes,
    validate_http_request,
    validate_language,
    validate_owned_audio_input,
    validate_source_file,
)
from trillim.components.stt._worker import transcribe_owned_audio_file
from trillim.errors import ProgressTimeoutError
from trillim.utils.filesystem import unlink_if_exists


async def _bounded_request_stream(request: Request):
    stream = request.stream().__aiter__()
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + UPLOAD_PROGRESS_TIMEOUT_SECONDS
    while True:
        now = loop.time()
        remaining = min(deadline - now, started + TOTAL_UPLOAD_TIMEOUT_SECONDS - now)
        if remaining <= 0:
            raise ProgressTimeoutError("audio upload timed out")
        try:
            async with asyncio.timeout(remaining):
                chunk = await anext(stream)
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise ProgressTimeoutError("audio upload timed out") from exc
        if not chunk:
            continue
        deadline = loop.time() + UPLOAD_PROGRESS_TIMEOUT_SECONDS
        yield chunk


class STT(Component):
    """Speech-to-text component with a fixed Phase 4 API."""

    def __init__(self) -> None:
        self._started = False
        self._admission = TranscriptionAdmission()
        self._spool_dir = Path(tempfile.gettempdir()) / "trillim-stt"
        self._active_task: asyncio.Task | None = None
        self._active_task_lock = asyncio.Lock()

    def router(self) -> APIRouter:
        """Return the STT HTTP router."""
        return build_router(self)

    async def start(self) -> None:
        """Verify STT dependencies are importable."""
        if self._started:
            return
        importlib.import_module("faster_whisper")
        self._started = True
        await self._admission.finish_starting()

    async def stop(self) -> None:
        """Drain admissions and cancel the active transcription, if any."""
        await self._admission.start_draining()
        active_task = await self._get_active_task()
        if active_task is not None and not active_task.done():
            active_task.cancel()
        await self._admission.wait_for_idle()
        self._started = False

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
    ) -> str:
        """Normalize in-memory audio bytes and return one final transcript."""
        self._require_started()
        normalized_language = validate_language(language)
        validated_bytes = validate_audio_bytes(audio_bytes)
        return await self._run_transcription(
            lambda: spool_audio_bytes(validated_bytes, spool_dir=self._spool_dir),
            language=normalized_language,
        )

    async def transcribe_file(
        self,
        path: str | Path,
        *,
        language: str | None = None,
    ) -> str:
        """Normalize a caller-owned file path and return one final transcript."""
        self._require_started()
        normalized_language = validate_language(language)
        source_path = validate_source_file(path)
        return await self._run_transcription(
            lambda: copy_source_file(source_path, spool_dir=self._spool_dir),
            language=normalized_language,
        )

    async def _transcribe_http_request(self, request: Request) -> str:
        self._require_started()
        validated_request = validate_http_request(
            content_type=request.headers.get("content-type"),
            content_length=request.headers.get("content-length"),
            language=request.query_params.get("language"),
        )
        return await self._run_transcription(
            lambda: spool_request_stream(
                _bounded_request_stream(request),
                spool_dir=self._spool_dir,
            ),
            language=validated_request.language,
        )

    async def _run_transcription(self, normalize_audio, *, language: str | None) -> str:
        async with await self._admission.acquire():
            async with self._track_active_task():
                owned_audio = None
                primary_error: BaseException | None = None
                try:
                    owned_audio = await normalize_audio()
                    validate_owned_audio_input(owned_audio)
                    return await transcribe_owned_audio_file(
                        owned_audio.path,
                        language=language,
                    )
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    if owned_audio is not None:
                        try:
                            unlink_if_exists(owned_audio.path)
                        except Exception:
                            if primary_error is None:
                                raise

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("STT is not started")

    async def _get_active_task(self) -> asyncio.Task | None:
        async with self._active_task_lock:
            return self._active_task

    @asynccontextmanager
    async def _track_active_task(self):
        task = asyncio.current_task()
        async with self._active_task_lock:
            self._active_task = task
        try:
            yield
        finally:
            async with self._active_task_lock:
                if self._active_task is task:
                    self._active_task = None
