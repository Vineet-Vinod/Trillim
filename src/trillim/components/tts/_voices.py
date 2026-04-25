"""Custom-voice persistence helpers for the TTS component."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from trillim.components.tts._limits import (
    MAX_CUSTOM_VOICES,
    MAX_TOTAL_CUSTOM_VOICE_BYTES,
    MAX_VOICE_STATE_BYTES,
    MAX_VOICE_UPLOAD_BYTES,
    VOICE_MANIFEST_NAME,
)
from trillim.components.tts._validation import (
    PayloadTooLargeError,
    load_safe_voice_state_bytes,
    load_safe_voice_state_safetensors,
    normalize_optional_name,
    normalize_required_name,
    open_validated_source_audio_file,
    save_voice_state_safetensors,
    validate_source_audio_path,
    validate_voice_bytes,
)
from trillim.errors import InvalidRequestError
from trillim.utils.filesystem import atomic_write_bytes, unlink_if_exists


logger = logging.getLogger(__name__)
VOICE_STATE_SUFFIX = ".safetensors"
LEGACY_VOICE_STATE_SUFFIX = ".state"


class VoiceStoreTamperedError(RuntimeError):
    """Raised when a requested custom-voice write/delete is unsafe."""


@dataclass(frozen=True, slots=True)
class ManagedVoiceEntry:
    """Persistent metadata for one managed custom voice."""

    name: str
    storage_id: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class OwnedVoiceUpload:
    """One Trillim-owned temporary voice upload."""

    path: Path
    size_bytes: int


async def load_custom_voice_states(
    root: Path,
    *,
    built_in_voice_names: tuple[str, ...],
) -> dict[str, dict]:
    """Discover and load valid persisted custom voices."""
    built_ins = set(built_in_voice_names)
    manifest = _load_manifest(root, built_ins=built_ins)
    loaded: dict[str, dict] = {}
    for name, entry in manifest.items():
        state_path = _state_path(root, entry.storage_id)
        state = _load_optional_state(entry, state_path)
        if state is not None:
            loaded[name] = state
    return loaded


async def publish_custom_voice(
    root: Path,
    *,
    name: str,
    upload: OwnedVoiceUpload,
    build_voice_state: Callable,
    existing_names: set[str],
) -> tuple[str, dict]:
    """Build, persist, and load one custom voice."""
    normalized_name = normalize_required_name(name, field_name="name")
    if normalized_name in existing_names:
        raise InvalidRequestError(f"voice '{normalized_name}' already exists")
    manifest = _load_manifest(root, built_ins=set())
    if normalized_name in manifest:
        raise InvalidRequestError(f"voice '{normalized_name}' already exists")
    if len(existing_names - set(manifest)) + len(manifest) >= MAX_CUSTOM_VOICES:
        raise InvalidRequestError(
            f"custom voice store already contains {MAX_CUSTOM_VOICES} voices"
        )

    state = load_safe_voice_state_bytes(await build_voice_state(upload.path))
    storage_id = _storage_id_for_name(normalized_name)
    final_path = _state_path(root, storage_id)
    temp_path = _create_temp_state_path(root)
    try:
        save_voice_state_safetensors(state, temp_path)
        loaded_state = load_safe_voice_state_safetensors(temp_path)
        size_bytes = temp_path.stat().st_size
        if size_bytes <= 0 or size_bytes > MAX_VOICE_STATE_BYTES:
            raise InvalidRequestError(
                f"voice state exceeds the {MAX_VOICE_STATE_BYTES} byte limit"
            )
        total_bytes = sum(entry.size_bytes for entry in manifest.values())
        if total_bytes + size_bytes > MAX_TOTAL_CUSTOM_VOICE_BYTES:
            raise InvalidRequestError(
                f"custom voice storage exceeds the {MAX_TOTAL_CUSTOM_VOICE_BYTES} byte limit"
            )

        _ensure_store_root(root)
        _raise_if_symlink_for_write(final_path)
        os.replace(temp_path, final_path)
        next_manifest = dict(manifest)
        next_manifest[normalized_name] = ManagedVoiceEntry(
            name=normalized_name,
            storage_id=storage_id,
            size_bytes=size_bytes,
        )
        try:
            _write_manifest(root, next_manifest)
        except Exception:
            unlink_if_exists(final_path)
            raise
    except Exception:
        unlink_if_exists(temp_path)
        raise
    return normalized_name, loaded_state


async def delete_custom_voice(root: Path, *, name: str) -> str:
    """Best-effort delete one custom voice from disk and manifest."""
    normalized_name = normalize_required_name(name, field_name="name")
    manifest = _load_manifest(root, built_ins=set())
    entry = manifest.get(normalized_name)
    if entry is None:
        return normalized_name
    state_path = _state_path(root, entry.storage_id)
    _raise_if_symlink_for_write(state_path)
    next_manifest = dict(manifest)
    del next_manifest[normalized_name]
    _write_manifest(root, next_manifest)
    unlink_if_exists(state_path)
    return normalized_name


def _load_manifest(root: Path, *, built_ins: set[str]) -> dict[str, ManagedVoiceEntry]:
    root = Path(root)
    manifest_path = root / VOICE_MANIFEST_NAME
    if not root.exists():
        return {}
    if _warn_if_symlink(root, "voice store root uses a symlink"):
        return {}
    if not root.is_dir():
        logger.warning("Skipping custom TTS voices: voice store root is malformed")
        return {}
    if not manifest_path.exists():
        _warn_for_legacy_files(root)
        if _has_non_legacy_children(root):
            logger.warning("Skipping custom TTS voices: voice manifest is missing")
        return {}
    if _warn_if_symlink(manifest_path, "voice manifest uses a symlink"):
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("Skipping custom TTS voices: voice manifest is malformed")
        return {}
    if not isinstance(payload, dict):
        logger.warning("Skipping custom TTS voices: voice manifest is malformed")
        return {}
    entries = payload.get("voices")
    if not isinstance(entries, list):
        logger.warning("Skipping custom TTS voices: voice manifest is malformed")
        return {}
    manifest: dict[str, ManagedVoiceEntry] = {}
    for item in entries:
        entry = _load_manifest_entry(item)
        if entry is None:
            continue
        if entry.name in manifest or entry.name in built_ins:
            logger.warning("Skipping malformed custom TTS voice manifest entry")
            continue
        manifest[entry.name] = entry
    _warn_for_inventory_mismatch(root, manifest)
    return manifest


def _write_manifest(root: Path, manifest: dict[str, ManagedVoiceEntry]) -> None:
    _ensure_store_root(root)
    manifest_path = root / VOICE_MANIFEST_NAME
    _raise_if_symlink_for_write(manifest_path)
    payload = {
        "voices": [
            {
                "name": entry.name,
                "storage_id": entry.storage_id,
                "size_bytes": entry.size_bytes,
            }
            for entry in sorted(manifest.values(), key=lambda item: item.name)
        ]
    }
    atomic_write_bytes(
        manifest_path,
        json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
    )


def _load_optional_state(entry: ManagedVoiceEntry, state_path: Path) -> dict | None:
    if _warn_if_symlink(state_path, f"voice file for {entry.name!r} uses a symlink"):
        return None
    try:
        stat_result = state_path.stat()
    except FileNotFoundError:
        _warn_skip(entry.name, "voice file is missing")
        return None
    except OSError:
        _warn_skip(entry.name, "voice file could not be read")
        return None
    if not state_path.is_file():
        _warn_skip(entry.name, "voice file is not a regular file")
        return None
    if stat_result.st_size != entry.size_bytes:
        _warn_skip(entry.name, "voice file size does not match manifest")
        return None
    try:
        return load_safe_voice_state_safetensors(state_path)
    except InvalidRequestError:
        _warn_skip(entry.name, "voice file is not valid safetensors")
        return None


def _load_manifest_entry(item: object) -> ManagedVoiceEntry | None:
    if not isinstance(item, dict):
        logger.warning("Skipping malformed custom TTS voice manifest entry")
        return None
    try:
        name = item["name"]
        storage_id = item["storage_id"]
        size_bytes = item["size_bytes"]
    except KeyError:
        logger.warning("Skipping malformed custom TTS voice manifest entry")
        return None
    try:
        normalized_name = normalize_optional_name(name, field_name="name")
    except InvalidRequestError:
        logger.warning("Skipping malformed custom TTS voice manifest entry")
        return None
    if (
        not isinstance(name, str)
        or not name
        or normalized_name != name
        or not isinstance(storage_id, str)
        or not isinstance(size_bytes, int)
    ):
        logger.warning("Skipping malformed custom TTS voice manifest entry")
        return None
    if size_bytes <= 0 or size_bytes > MAX_VOICE_STATE_BYTES:
        logger.warning("Skipping malformed custom TTS voice manifest entry")
        return None
    if storage_id != _storage_id_for_name(name):
        logger.warning("Skipping malformed custom TTS voice manifest entry")
        return None
    return ManagedVoiceEntry(name=name, storage_id=storage_id, size_bytes=size_bytes)


def _warn_for_inventory_mismatch(
    root: Path,
    manifest: dict[str, ManagedVoiceEntry],
) -> None:
    expected_names = {VOICE_MANIFEST_NAME}
    expected_names.update(
        f"{entry.storage_id}{VOICE_STATE_SUFFIX}" for entry in manifest.values()
    )
    unexpected: list[str] = []
    try:
        children = list(root.iterdir())
    except OSError:
        logger.warning("Skipping custom TTS voice inventory check: voice store root is malformed")
        return
    for child in children:
        if _warn_if_symlink(child, "voice store child uses a symlink"):
            continue
        if child.name in expected_names:
            continue
        if child.name.endswith(LEGACY_VOICE_STATE_SUFFIX):
            logger.warning("Skipping legacy TTS voice file: %s", child)
            continue
        unexpected.append(child.name)
    if unexpected:
        logger.warning(
            "Skipping unexpected TTS voice store files: %s",
            ", ".join(sorted(repr(name) for name in unexpected)),
        )


def _warn_for_legacy_files(root: Path) -> None:
    try:
        children = list(root.iterdir())
    except OSError:
        logger.warning("Skipping custom TTS voices: voice store root is malformed")
        return
    for child in children:
        if _warn_if_symlink(child, "voice store child uses a symlink"):
            continue
        if child.name.endswith(LEGACY_VOICE_STATE_SUFFIX):
            logger.warning("Skipping legacy TTS voice file: %s", child)


def _has_non_legacy_children(root: Path) -> bool:
    try:
        return any(
            child.name != VOICE_MANIFEST_NAME
            and not child.name.endswith(LEGACY_VOICE_STATE_SUFFIX)
            for child in root.iterdir()
        )
    except OSError:
        logger.warning("Skipping custom TTS voices: voice store root is malformed")
        return False


def _ensure_store_root(root: Path) -> None:
    if root.exists():
        _raise_if_symlink_for_write(root)
        if not root.is_dir():
            raise VoiceStoreTamperedError("voice store root is malformed")
    root.mkdir(parents=True, exist_ok=True)


def _state_path(root: Path, storage_id: str) -> Path:
    return root / f"{storage_id}{VOICE_STATE_SUFFIX}"


def _create_temp_state_path(root: Path) -> Path:
    _ensure_store_root(root)
    fd, temp_name = tempfile.mkstemp(
        dir=root,
        prefix="voice-",
        suffix=VOICE_STATE_SUFFIX,
    )
    os.close(fd)
    return Path(temp_name)


def _warn_if_symlink(path: Path, reason: str) -> bool:
    if path.is_symlink():
        logger.warning("Skipping custom TTS voice store path %s: %s", path, reason)
        return True
    return False


def _raise_if_symlink_for_write(path: Path) -> None:
    if path.is_symlink():
        raise VoiceStoreTamperedError("voice store must not use symlinks")


def _warn_skip(voice_name: str, reason: str) -> None:
    logger.warning("Skipping custom TTS voice %r: %s", voice_name, reason)


async def spool_request_voice_stream(
    chunks: AsyncIterator[bytes],
    *,
    spool_dir: Path,
) -> OwnedVoiceUpload:
    """Copy one async voice-upload stream into Trillim-owned temp storage."""
    fd, temp_path = _create_owned_temp_file(spool_dir)
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            async for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_VOICE_UPLOAD_BYTES:
                    raise PayloadTooLargeError(
                        f"voice upload exceeds the {MAX_VOICE_UPLOAD_BYTES} byte limit"
                    )
                handle.write(chunk)
        if total <= 0:
            raise InvalidRequestError("audio must not be empty")
        return OwnedVoiceUpload(path=temp_path, size_bytes=total)
    except BaseException:
        unlink_if_exists(temp_path)
        raise


async def spool_voice_bytes(
    audio_bytes: bytes,
    *,
    spool_dir: Path,
) -> OwnedVoiceUpload:
    """Copy in-memory voice bytes into Trillim-owned temp storage."""
    validated_bytes = validate_voice_bytes(audio_bytes)
    fd, temp_path = _create_owned_temp_file(spool_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(validated_bytes)
        return OwnedVoiceUpload(path=temp_path, size_bytes=len(validated_bytes))
    except BaseException:
        unlink_if_exists(temp_path)
        raise


async def copy_source_audio(
    path: str | Path,
    *,
    spool_dir: Path,
) -> OwnedVoiceUpload:
    """Copy one caller-owned audio file into Trillim-owned temp storage."""
    source_path = validate_source_audio_path(path)
    source_fd = open_validated_source_audio_file(source_path)
    return await asyncio.to_thread(_copy_source_audio_sync, source_fd, spool_dir)


def _copy_source_audio_sync(source_fd: int, spool_dir: Path) -> OwnedVoiceUpload:
    fd, temp_path = _create_owned_temp_file(spool_dir)
    total = 0
    raw_source_fd = source_fd
    raw_temp_fd = fd
    try:
        with os.fdopen(raw_source_fd, "rb") as source_handle:
            raw_source_fd = -1
            with os.fdopen(raw_temp_fd, "wb") as temp_handle:
                raw_temp_fd = -1
                while True:
                    chunk = source_handle.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_VOICE_UPLOAD_BYTES:
                        raise PayloadTooLargeError(
                            f"voice upload exceeds the {MAX_VOICE_UPLOAD_BYTES} byte limit"
                        )
                    temp_handle.write(chunk)
        return OwnedVoiceUpload(path=temp_path, size_bytes=total)
    except BaseException:
        if raw_source_fd >= 0:
            os.close(raw_source_fd)
        if raw_temp_fd >= 0:
            os.close(raw_temp_fd)
        unlink_if_exists(temp_path)
        raise


def _create_owned_temp_file(
    spool_dir: Path,
    *,
    suffix: str = ".audio",
) -> tuple[int, Path]:
    spool_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=spool_dir,
        prefix="tts-",
        suffix=suffix,
    )
    return fd, Path(temp_name)


def _storage_id_for_name(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]
