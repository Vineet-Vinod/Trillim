"""Validation helpers for Phase 4 STT inputs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from trillim.components.stt._config import OwnedAudioInput, SourceFileSnapshot
from trillim.components.stt._limits import MAX_LANGUAGE_CHARS, MAX_UPLOAD_BYTES
from trillim.errors import InvalidRequestError

_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z]{2,8})*$")


class PayloadTooLargeError(InvalidRequestError):
    """Raised when an audio payload exceeds the fixed Phase 4 byte cap."""


@dataclass(frozen=True, slots=True)
class HTTPTranscriptionRequest:
    """Validated HTTP request metadata for STT."""

    content_type: str
    content_length: int | None
    language: str | None


def validate_language(language: str | None) -> str | None:
    """Validate the optional STT language hint."""
    if language is None:
        return None
    value = language.strip()
    if not value:
        raise InvalidRequestError("language must not be empty")
    if len(value) > MAX_LANGUAGE_CHARS:
        raise InvalidRequestError(
            f"language exceeds the {MAX_LANGUAGE_CHARS} character limit"
        )
    if _LANGUAGE_RE.fullmatch(value) is None:
        raise InvalidRequestError("language must contain only letters and hyphens")
    return value.lower()


def validate_audio_bytes(audio_bytes: bytes) -> bytes:
    """Validate SDK byte input before spooling."""
    if not isinstance(audio_bytes, bytes):
        raise InvalidRequestError("audio_bytes must be bytes")
    if not audio_bytes:
        raise InvalidRequestError("audio_bytes must not be empty")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit"
        )
    return audio_bytes


def validate_source_file(path: str | Path) -> Path:
    """Validate a caller-owned source path before normalization."""
    if not str(path):
        raise InvalidRequestError("path is required")
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise InvalidRequestError(f"audio file does not exist: {resolved}")
    if not resolved.is_file():
        raise InvalidRequestError(f"audio file is not a regular file: {resolved}")
    if resolved.stat().st_size > MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit"
        )
    return resolved


def validate_owned_audio_input(owned_audio: OwnedAudioInput) -> OwnedAudioInput:
    """Validate the Trillim-owned normalized copy before worker launch."""
    if owned_audio.size_bytes <= 0:
        raise InvalidRequestError("audio input must not be empty")
    if owned_audio.size_bytes > MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit"
        )
    return owned_audio


def validate_http_request(
    *,
    content_type: str | None,
    content_length: str | None,
    language: str | None,
) -> HTTPTranscriptionRequest:
    """Validate the Phase 4 raw-body HTTP request contract."""
    normalized_content_type = _normalize_content_type(content_type)
    normalized_length = _validate_content_length(content_length)
    normalized_language = validate_language(language)
    return HTTPTranscriptionRequest(
        content_type=normalized_content_type,
        content_length=normalized_length,
        language=normalized_language,
    )


def snapshot_source_file(stat_result: os.stat_result) -> SourceFileSnapshot:
    """Capture the metadata we use for best-effort source mutation detection."""
    return SourceFileSnapshot(
        size_bytes=stat_result.st_size,
        modified_ns=stat_result.st_mtime_ns,
    )


def validate_source_snapshot(
    before: SourceFileSnapshot,
    after: SourceFileSnapshot,
) -> None:
    """Reject caller-owned source files that changed across the copy."""
    if before != after:
        raise InvalidRequestError("audio file changed while it was being copied")


def _normalize_content_type(content_type: str | None) -> str:
    if content_type is None:
        raise InvalidRequestError(
            "content-type must be audio/* or application/octet-stream"
        )
    value = content_type.split(";", 1)[0].strip().lower()
    if value == "application/octet-stream" or value.startswith("audio/"):
        return value
    raise InvalidRequestError(
        "content-type must be audio/* or application/octet-stream"
    )


def _validate_content_length(content_length: str | None) -> int | None:
    if content_length is None:
        return None
    try:
        parsed = int(content_length)
    except ValueError as exc:
        raise InvalidRequestError("invalid content-length header") from exc
    if parsed < 0:
        raise InvalidRequestError("invalid content-length header")
    if parsed > MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"audio input exceeds the {MAX_UPLOAD_BYTES} byte limit"
        )
    return parsed
