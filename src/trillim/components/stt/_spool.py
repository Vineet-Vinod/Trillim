"""Owned-temp normalization helpers for Phase 4 STT."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path

from trillim.components.stt._config import OwnedAudioInput
from trillim.components.stt._limits import MAX_UPLOAD_BYTES, SPOOL_CHUNK_SIZE_BYTES
from trillim.components.stt._validation import (
    PayloadTooLargeError,
    open_validated_source_file,
    snapshot_source_file,
    validate_source_snapshot,
)
from trillim.utils.filesystem import unlink_if_exists


class _SourceCopyCancelledError(Exception):
    """Internal sentinel used to stop a background source-file copy cleanly."""


async def spool_request_stream(
    chunks: AsyncIterator[bytes],
    *,
    spool_dir: Path,
) -> OwnedAudioInput:
    """Copy an async raw-body stream into Trillim-owned temp storage."""
    fd, temp_path = _create_owned_temp_file(spool_dir)
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            async for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise PayloadTooLargeError(
                        f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit"
                    )
                handle.write(chunk)
        return OwnedAudioInput(path=temp_path, size_bytes=total)
    except BaseException:
        unlink_if_exists(temp_path)
        raise


async def spool_audio_bytes(
    audio_bytes: bytes,
    *,
    spool_dir: Path,
) -> OwnedAudioInput:
    """Copy SDK byte input into Trillim-owned temp storage."""
    fd, temp_path = _create_owned_temp_file(spool_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(audio_bytes)
        return OwnedAudioInput(path=temp_path, size_bytes=len(audio_bytes))
    except BaseException:
        unlink_if_exists(temp_path)
        raise


async def copy_source_file(
    source_path: Path,
    *,
    spool_dir: Path,
) -> OwnedAudioInput:
    """Normalize a caller-owned path into a Trillim-owned temp file."""
    source_fd = open_validated_source_file(source_path)
    cancel_event = threading.Event()
    copy_task = asyncio.create_task(
        asyncio.to_thread(
            _copy_source_file_sync,
            source_fd,
            spool_dir,
            cancel_event,
        )
    )
    try:
        return await asyncio.shield(copy_task)
    except asyncio.CancelledError:
        cancel_event.set()
        try:
            owned_audio = await asyncio.shield(copy_task)
        except _SourceCopyCancelledError:
            pass
        except Exception:
            pass
        else:
            unlink_if_exists(owned_audio.path)
        raise


def _copy_source_file_sync(
    source_fd: int,
    spool_dir: Path,
    cancel_event: threading.Event | None = None,
) -> OwnedAudioInput:
    raw_source_fd = source_fd
    raw_temp_fd = -1
    temp_path: Path | None = None
    total = 0
    try:
        temp_fd, temp_path = _create_owned_temp_file(spool_dir)
        raw_temp_fd = temp_fd
        with os.fdopen(raw_source_fd, "rb") as source_handle:
            raw_source_fd = -1
            with os.fdopen(raw_temp_fd, "wb") as temp_handle:
                raw_temp_fd = -1
                before = snapshot_source_file(os.fstat(source_handle.fileno()))
                while True:
                    _raise_if_copy_cancelled(cancel_event)
                    chunk = _read_source_chunk(source_handle)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise PayloadTooLargeError(
                            f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit"
                        )
                    _raise_if_copy_cancelled(cancel_event)
                    temp_handle.write(chunk)
                after = snapshot_source_file(os.fstat(source_handle.fileno()))
        validate_source_snapshot(before, after)
        return OwnedAudioInput(path=temp_path, size_bytes=total)
    except BaseException:
        if raw_source_fd >= 0:
            os.close(raw_source_fd)
        if raw_temp_fd >= 0:
            os.close(raw_temp_fd)
        if temp_path is not None:
            unlink_if_exists(temp_path)
        raise


def _read_source_chunk(source_handle) -> bytes:
    return source_handle.read(SPOOL_CHUNK_SIZE_BYTES)


def _raise_if_copy_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _SourceCopyCancelledError()


def _create_owned_temp_file(spool_dir: Path) -> tuple[int, Path]:
    spool_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=spool_dir,
        prefix="stt-",
        suffix=".audio",
    )
    return fd, Path(temp_name)
