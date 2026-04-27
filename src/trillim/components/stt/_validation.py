from __future__ import annotations

import re
from dataclasses import dataclass

from trillim.components.stt._limits import MAX_LANGUAGE_CHARS, MAX_UPLOAD_BYTES
from trillim.errors import InvalidRequestError

_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z]{2,8})*$")


class PayloadTooLargeError(InvalidRequestError):
    pass


@dataclass(frozen=True, slots=True)
class HTTPTranscriptionRequest:
    content_length: int | None
    language: str | None


def validate_http_request(
    *,
    content_type: str | None,
    content_length: str | None,
    language: str | None,
) -> HTTPTranscriptionRequest:
    _normalize_content_type(content_type)
    return HTTPTranscriptionRequest(
        content_length=_validate_content_length(content_length),
        language=_validate_language(language),
    )


def _validate_language(language: str | None) -> str | None:
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


def _normalize_content_type(content_type: str | None) -> str:
    if content_type is None:
        raise InvalidRequestError("content-type must be audio/wav or application/octet-stream")
    value = content_type.split(";", 1)[0].strip().lower()
    if value in {"application/octet-stream", "audio/wav", "audio/x-wav"}:
        return value
    raise InvalidRequestError("content-type must be audio/wav or application/octet-stream")


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
