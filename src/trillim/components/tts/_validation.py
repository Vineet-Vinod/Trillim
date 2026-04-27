"""Validation helpers for the TTS component."""

from __future__ import annotations

import errno
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trillim.components.tts._limits import (
    DEFAULT_SPEED,
    MAX_HTTP_TEXT_BYTES,
    MAX_INPUT_TEXT_CHARS,
    MAX_SPEED,
    MAX_VOICE_STATE_BYTES,
    MAX_VOICE_UPLOAD_BYTES,
    MIN_SPEED,
)
from trillim.errors import InvalidRequestError

_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9]+$")


class PayloadTooLargeError(InvalidRequestError):
    """Raised when a bounded TTS payload exceeds its byte cap."""


@dataclass(frozen=True, slots=True)
class HTTPSpeechRequest:
    """Validated HTTP metadata for one speech request."""

    content_length: int | None
    voice: str | None
    speed: float


@dataclass(frozen=True, slots=True)
class HTTPVoiceUploadRequest:
    """Validated HTTP metadata for one custom-voice upload."""

    content_length: int | None
    name: str


def validate_text(text: str) -> str:
    """Validate one SDK speech input."""
    if not isinstance(text, str):
        raise InvalidRequestError("text must be a string")
    if len(text) > MAX_INPUT_TEXT_CHARS:
        raise InvalidRequestError(
            f"text exceeds the {MAX_INPUT_TEXT_CHARS} character limit"
        )
    if not text.strip():
        raise InvalidRequestError("text must not be empty")
    return text


def validate_http_speech_request(
    *,
    content_length: str | None,
    voice: str | None,
    speed: str | None,
    default_speed: float = DEFAULT_SPEED,
) -> HTTPSpeechRequest:
    """Validate the metadata for the raw-body HTTP speech route."""
    return HTTPSpeechRequest(
        content_length=_validate_content_length(
            content_length,
            limit=MAX_HTTP_TEXT_BYTES,
            kind="speech input",
        ),
        voice=normalize_optional_name(voice, field_name="voice"),
        speed=validate_speed(default_speed if speed is None else speed),
    )


def validate_http_voice_upload_request(
    *,
    content_length: str | None,
    name: str | None,
) -> HTTPVoiceUploadRequest:
    """Validate the metadata for the raw-body HTTP voice-upload route."""
    return HTTPVoiceUploadRequest(
        content_length=_validate_content_length(
            content_length,
            limit=MAX_VOICE_UPLOAD_BYTES,
            kind="voice upload",
        ),
        name=normalize_required_name(name, field_name="name"),
    )


def validate_http_speech_body(body: bytes) -> str:
    """Validate and decode one raw-body speech payload."""
    if len(body) > MAX_HTTP_TEXT_BYTES:
        raise PayloadTooLargeError(
            f"speech input exceeds the {MAX_HTTP_TEXT_BYTES} byte limit"
        )
    if not body:
        raise InvalidRequestError("speech input must not be empty")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidRequestError("speech input must be valid UTF-8") from exc
    return validate_text(text)


def validate_voice_bytes(audio_bytes: bytes) -> bytes:
    """Validate in-memory voice-upload bytes."""
    if not isinstance(audio_bytes, bytes):
        raise InvalidRequestError("audio must be bytes")
    if not audio_bytes:
        raise InvalidRequestError("audio must not be empty")
    if len(audio_bytes) > MAX_VOICE_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            f"voice upload exceeds the {MAX_VOICE_UPLOAD_BYTES} byte limit"
        )
    return audio_bytes


def validate_speed(speed: float | str) -> float:
    """Validate one speed value."""
    try:
        value = float(speed)
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError("speed must be a number") from exc
    if not (MIN_SPEED <= value <= MAX_SPEED):
        raise InvalidRequestError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
    return value


def normalize_required_name(name: str | None, *, field_name: str) -> str:
    """Normalize one required name field."""
    if name is None:
        raise InvalidRequestError(f"{field_name} header is required")
    value = normalize_optional_name(name, field_name=field_name)
    if value is None:
        raise InvalidRequestError(f"{field_name} must not be empty")
    return value


def normalize_optional_name(name: str | None, *, field_name: str) -> str | None:
    """Normalize one optional name-like field."""
    if name is None:
        return None
    value = str(name).strip()
    if not value:
        raise InvalidRequestError(f"{field_name} must not be empty")
    if _VOICE_NAME_RE.fullmatch(value) is None:
        raise InvalidRequestError(
            f"{field_name} must contain only letters and digits"
        )
    return value


def validate_voice_state_bytes(state_bytes: bytes) -> bytes:
    """Validate serialized custom-voice state emitted by the worker."""
    if not state_bytes:
        raise InvalidRequestError("voice state must not be empty")
    if len(state_bytes) > MAX_VOICE_STATE_BYTES:
        raise InvalidRequestError(
            f"voice state exceeds the {MAX_VOICE_STATE_BYTES} byte limit"
        )
    return state_bytes


def load_safe_voice_state_safetensors(path: str | Path):
    """Safely load one bounded Pocket TTS voice state from safetensors."""
    import torch
    from safetensors.torch import load_file

    voice_path = Path(path)
    try:
        stat_result = voice_path.stat()
    except OSError as exc:
        raise InvalidRequestError("voice state is malformed") from exc
    if stat_result.st_size <= 0 or stat_result.st_size > MAX_VOICE_STATE_BYTES:
        raise InvalidRequestError("voice state is malformed")
    try:
        flat_state = load_file(str(voice_path), device="cpu")
    except Exception as exc:
        raise InvalidRequestError("voice state is malformed") from exc
    state = _unflatten_safetensors_voice_state(flat_state, torch)
    _validate_loaded_voice_state_value(state, torch)
    return state


def load_safe_voice_state_safetensors_bytes(state_bytes: bytes):
    """Safely load one bounded Pocket TTS voice state from safetensors bytes."""
    payload = validate_voice_state_bytes(state_bytes)
    fd, temp_name = tempfile.mkstemp(suffix=".safetensors")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        return load_safe_voice_state_safetensors(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def save_voice_state_safetensors(state: dict, path: str | Path) -> None:
    """Save one Pocket TTS voice state in safetensors format."""
    import torch
    from safetensors.torch import save_file

    if not isinstance(state, dict) or not state:
        raise InvalidRequestError("voice state is malformed")
    _validate_loaded_voice_state_value(state, torch)
    flat_state = _flatten_safetensors_voice_state(state, torch)
    try:
        save_file(flat_state, str(path), metadata={"format": "trillim-pocket-tts-state"})
    except Exception as exc:
        raise InvalidRequestError("voice state is malformed") from exc


def dump_voice_state_safetensors_bytes(state: dict) -> bytes:
    """Serialize one Pocket TTS voice state to safetensors bytes."""
    fd, temp_name = tempfile.mkstemp(suffix=".safetensors")
    temp_path = Path(temp_name)
    os.close(fd)
    try:
        save_voice_state_safetensors(state, temp_path)
        return validate_voice_state_bytes(temp_path.read_bytes())
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def validate_source_audio_path(path: str | Path) -> Path:
    """Perform cheap preliminary validation on one caller-owned audio path."""
    if isinstance(path, str):
        if not path:
            raise InvalidRequestError("path is required")
    elif getattr(path, "_raw_paths", None) == [""]:
        raise InvalidRequestError("path is required")
    return Path(path).expanduser()


def open_validated_source_audio_file(path: Path) -> int:
    """Open one source audio file while rejecting a symlinked final component."""
    try:
        if path.is_symlink():
            raise InvalidRequestError(f"audio file must not use symlinks: {path}")
        path_stat = os.stat(path)
    except FileNotFoundError as exc:
        raise InvalidRequestError(f"audio file does not exist: {path}") from exc
    except OSError as exc:
        raise InvalidRequestError(f"audio file could not be opened: {path}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise InvalidRequestError(f"audio file is not a regular file: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        raise InvalidRequestError(f"audio file does not exist: {path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise InvalidRequestError(f"audio file must not use symlinks: {path}") from exc
        raise InvalidRequestError(f"audio file could not be opened: {path}") from exc
    try:
        stat_result = os.fstat(fd)
        if not stat.S_ISREG(stat_result.st_mode):
            raise InvalidRequestError(f"audio file is not a regular file: {path}")
        if stat_result.st_size > MAX_VOICE_UPLOAD_BYTES:
            raise PayloadTooLargeError(
                f"voice upload exceeds the {MAX_VOICE_UPLOAD_BYTES} byte limit"
            )
        if stat_result.st_size <= 0:
            raise InvalidRequestError("audio file must not be empty")
        return fd
    except Exception:
        os.close(fd)
        raise


def _validate_content_length(
    content_length: str | None,
    *,
    limit: int,
    kind: str,
) -> int | None:
    if content_length is None:
        return None
    try:
        parsed = int(content_length)
    except ValueError as exc:
        raise InvalidRequestError("invalid content-length header") from exc
    if parsed < 0:
        raise InvalidRequestError("invalid content-length header")
    if parsed > limit:
        raise PayloadTooLargeError(f"{kind} exceeds the {limit} byte limit")
    return parsed


def _validate_loaded_voice_state_value(value, torch) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise InvalidRequestError("voice state is malformed")
            _validate_loaded_voice_state_value(child, torch)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_loaded_voice_state_value(child, torch)
        return
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, (str, int, float, bool, type(None))):
        return
    raise InvalidRequestError("voice state is malformed")


def _flatten_safetensors_voice_state(state: dict, torch) -> dict[str, object]:
    flat_state: dict[str, object] = {}
    for module_name, module_state in state.items():
        if not isinstance(module_name, str) or "/" in module_name:
            raise InvalidRequestError("voice state is malformed")
        if not isinstance(module_state, dict):
            raise InvalidRequestError("voice state is malformed")
        for key, tensor in module_state.items():
            if not isinstance(key, str) or "/" in key:
                raise InvalidRequestError("voice state is malformed")
            if not torch.is_tensor(tensor):
                raise InvalidRequestError("voice state is malformed")
            flat_state[f"{module_name}/{key}"] = tensor.detach().cpu().contiguous()
    if not flat_state:
        raise InvalidRequestError("voice state is malformed")
    return flat_state


def _unflatten_safetensors_voice_state(flat_state: dict, torch) -> dict:
    if not isinstance(flat_state, dict) or not flat_state:
        raise InvalidRequestError("voice state is malformed")
    state: dict[str, dict[str, object]] = {}
    for flat_key, tensor in flat_state.items():
        if not isinstance(flat_key, str) or "/" not in flat_key:
            raise InvalidRequestError("voice state is malformed")
        module_name, key = flat_key.split("/", 1)
        if not module_name or not key or not torch.is_tensor(tensor):
            raise InvalidRequestError("voice state is malformed")
        state.setdefault(module_name, {})[key] = tensor
    return state
