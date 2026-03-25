"""Managed custom-voice storage for the TTS component."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import AsyncIterator
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
    open_validated_source_audio_file,
    validate_source_audio_path,
    validate_voice_bytes,
    validate_voice_state_bytes,
)
from trillim.errors import InvalidRequestError
from trillim.utils.filesystem import atomic_write_bytes, unlink_if_exists


class VoiceStoreTamperedError(RuntimeError):
    """Raised when managed custom-voice files fail closed."""


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


@dataclass(frozen=True, slots=True)
class ResolvedVoice:
    """One voice reference ready for a synthesis worker."""

    name: str
    kind: str
    reference: str
    cleanup_path: Path | None = None


class VoiceStore:
    """Own the managed voice directory and manifest."""

    def __init__(self, root: Path, *, built_in_voice_names: tuple[str, ...]) -> None:
        self._root = Path(root)
        self._manifest_path = self._root / VOICE_MANIFEST_NAME
        self._built_ins = tuple(built_in_voice_names)
        self._lock = asyncio.Lock()
        self._tamper_error: VoiceStoreTamperedError | None = None

    async def list_names(self) -> list[str]:
        """Return built-in voices followed by custom voices."""
        async with self._lock:
            manifest = self._load_manifest_locked()
            custom_names = sorted(manifest)
            return [*self._built_ins, *custom_names]

    async def ensure_name_available(self, name: str) -> None:
        """Fail if the requested custom voice name already exists."""
        async with self._lock:
            self._ensure_name_available_locked(name, self._load_manifest_locked())

    async def register_owned_upload(
        self,
        *,
        name: str,
        upload: OwnedVoiceUpload,
        build_voice_state,
    ) -> str:
        """Build and publish one custom voice from an owned upload path."""
        async with self._lock:
            manifest = self._load_manifest_locked()
            self._ensure_name_available_locked(name, manifest)
            state_bytes = validate_voice_state_bytes(
                await build_voice_state(upload.path)
            )
            try:
                load_safe_voice_state_bytes(state_bytes)
            except InvalidRequestError as exc:
                raise RuntimeError("PocketTTS returned malformed voice state") from exc
            if len(manifest) >= MAX_CUSTOM_VOICES:
                raise InvalidRequestError(
                    f"custom voice store already contains {MAX_CUSTOM_VOICES} voices"
                )
            total_bytes = sum(entry.size_bytes for entry in manifest.values())
            if total_bytes + len(state_bytes) > MAX_TOTAL_CUSTOM_VOICE_BYTES:
                raise InvalidRequestError(
                    f"custom voice storage exceeds the {MAX_TOTAL_CUSTOM_VOICE_BYTES} byte limit"
                )
            storage_id = _storage_id_for_name(name)
            entry = ManagedVoiceEntry(
                name=name,
                storage_id=storage_id,
                size_bytes=len(state_bytes),
            )
            state_path = self._state_path(storage_id)
            self._ensure_store_root_locked()
            self._raise_if_managed_symlink_locked(state_path)
            atomic_write_bytes(state_path, state_bytes)
            next_manifest = dict(manifest)
            next_manifest[name] = entry
            try:
                self._write_manifest_locked(next_manifest)
            except Exception:
                unlink_if_exists(state_path)
                raise
            return name

    async def delete(self, name: str, *, protected_name: str | None = None) -> str:
        """Delete one managed custom voice by name."""
        async with self._lock:
            if name in self._built_ins:
                raise InvalidRequestError(
                    f"voice '{name}' is built in and cannot be deleted"
                )
            if protected_name is not None and name == protected_name:
                raise InvalidRequestError(
                    f"voice '{name}' is currently in use as default_voice"
                )
            manifest = self._load_manifest_locked()
            entry = manifest.get(name)
            if entry is None:
                raise KeyError(name)
            state_path = self._state_path(entry.storage_id)
            self._raise_if_managed_symlink_locked(state_path)
            state_bytes = state_path.read_bytes()
            next_manifest = dict(manifest)
            del next_manifest[name]
            unlink_if_exists(state_path)
            try:
                self._write_manifest_locked(next_manifest)
            except Exception:
                atomic_write_bytes(state_path, state_bytes)
                raise
            return name

    async def resolve_for_session(
        self,
        name: str,
        *,
        spool_dir: Path,
    ) -> ResolvedVoice:
        """Resolve one voice name into a worker-ready reference."""
        async with self._lock:
            if name in self._built_ins:
                return ResolvedVoice(name=name, kind="predefined", reference=name)
            manifest = self._load_manifest_locked()
            entry = manifest.get(name)
            if entry is None:
                raise InvalidRequestError(f"unknown voice: {name}")
            state_bytes = self._load_state_bytes_locked(entry)
        temp_path = await spool_voice_state_bytes(state_bytes, spool_dir=spool_dir)
        return ResolvedVoice(
            name=name,
            kind="state_file",
            reference=str(temp_path),
            cleanup_path=temp_path,
        )

    def _load_manifest_locked(self) -> dict[str, ManagedVoiceEntry]:
        self._raise_if_store_tampered_locked()
        if not self._root.exists():
            return {}
        self._raise_if_managed_symlink_locked(self._root)
        if not self._root.is_dir():
            self._mark_store_tampered_locked("voice store root is malformed")
        if not self._manifest_path.exists():
            try:
                has_entries = next(self._root.iterdir(), None) is not None
            except OSError:
                self._mark_store_tampered_locked("voice store root is malformed")
            if has_entries:
                self._mark_store_tampered_locked("voice manifest is missing")
            return {}
        self._raise_if_managed_symlink_locked(self._manifest_path)
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            self._mark_store_tampered_locked("voice manifest is malformed")
        if not isinstance(payload, dict):
            self._mark_store_tampered_locked("voice manifest is malformed")
        entries = payload.get("voices")
        if not isinstance(entries, list):
            self._mark_store_tampered_locked("voice manifest is malformed")
        manifest: dict[str, ManagedVoiceEntry] = {}
        for item in entries:
            entry = self._load_manifest_entry_locked(item)
            if entry.name in manifest or entry.name in self._built_ins:
                self._mark_store_tampered_locked("voice manifest is malformed")
            self._validate_manifest_state_file_locked(entry)
            manifest[entry.name] = entry
        self._validate_store_inventory_locked(manifest)
        return manifest

    def _write_manifest_locked(
        self,
        manifest: dict[str, ManagedVoiceEntry],
    ) -> None:
        self._ensure_store_root_locked()
        self._raise_if_managed_symlink_locked(self._manifest_path)
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
            self._manifest_path,
            json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
        )

    def _ensure_name_available_locked(
        self,
        name: str,
        manifest: dict[str, ManagedVoiceEntry],
    ) -> None:
        if name in self._built_ins or name in manifest:
            raise InvalidRequestError(f"voice '{name}' already exists")

    def _load_state_bytes_locked(self, entry: ManagedVoiceEntry) -> bytes:
        state_path = self._state_path(entry.storage_id)
        self._raise_if_managed_symlink_locked(state_path)
        try:
            state_bytes = validate_voice_state_bytes(state_path.read_bytes())
        except (InvalidRequestError, OSError):
            self._mark_store_tampered_locked(
                f"custom voice state for '{entry.name}' is malformed"
            )
        if len(state_bytes) != entry.size_bytes:
            self._mark_store_tampered_locked(
                f"custom voice state for '{entry.name}' is malformed"
            )
        try:
            load_safe_voice_state_bytes(state_bytes)
        except InvalidRequestError:
            self._mark_store_tampered_locked(
                f"custom voice state for '{entry.name}' is malformed"
            )
        return state_bytes

    def _ensure_store_root_locked(self) -> None:
        if self._root.exists():
            self._raise_if_managed_symlink_locked(self._root)
            if not self._root.is_dir():
                self._mark_store_tampered_locked("voice store root is malformed")
        self._root.mkdir(parents=True, exist_ok=True)

    def _state_path(self, storage_id: str) -> Path:
        return self._root / f"{storage_id}.state"

    def _raise_if_symlink(self, path: Path, message: str) -> None:
        if path.is_symlink():
            raise RuntimeError(message)

    def _load_manifest_entry_locked(self, item: object) -> ManagedVoiceEntry:
        if not isinstance(item, dict):
            self._mark_store_tampered_locked("voice manifest is malformed")
        try:
            name = item["name"]
            storage_id = item["storage_id"]
            size_bytes = item["size_bytes"]
        except KeyError:
            self._mark_store_tampered_locked("voice manifest is malformed")
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or not isinstance(storage_id, str)
            or not isinstance(size_bytes, int)
        ):
            self._mark_store_tampered_locked("voice manifest is malformed")
        if size_bytes <= 0 or size_bytes > MAX_VOICE_STATE_BYTES:
            self._mark_store_tampered_locked("voice manifest is malformed")
        if storage_id != _storage_id_for_name(name):
            self._mark_store_tampered_locked("voice manifest is malformed")
        return ManagedVoiceEntry(
            name=name,
            storage_id=storage_id,
            size_bytes=size_bytes,
        )

    def _validate_manifest_state_file_locked(self, entry: ManagedVoiceEntry) -> None:
        state_path = self._state_path(entry.storage_id)
        self._raise_if_managed_symlink_locked(state_path)
        try:
            stat_result = state_path.stat()
        except FileNotFoundError:
            self._mark_store_tampered_locked(
                f"custom voice state for '{entry.name}' is missing"
            )
        except OSError:
            self._mark_store_tampered_locked(
                f"custom voice state for '{entry.name}' is malformed"
            )
        if not state_path.is_file():
            self._mark_store_tampered_locked(
                f"custom voice state for '{entry.name}' is malformed"
            )
        if stat_result.st_size != entry.size_bytes:
            self._mark_store_tampered_locked(
                f"custom voice state for '{entry.name}' is malformed"
            )

    def _validate_store_inventory_locked(
        self,
        manifest: dict[str, ManagedVoiceEntry],
    ) -> None:
        expected_names = {VOICE_MANIFEST_NAME}
        expected_names.update(
            f"{entry.storage_id}.state" for entry in manifest.values()
        )
        unexpected: list[str] = []
        try:
            children = list(self._root.iterdir())
        except OSError:
            self._mark_store_tampered_locked("voice store root is malformed")
        for child in children:
            self._raise_if_managed_symlink_locked(child)
            if child.name not in expected_names:
                unexpected.append(child.name)
        if unexpected:
            unexpected_list = ", ".join(sorted(repr(name) for name in unexpected))
            self._mark_store_tampered_locked(
                "voice store contains unexpected files: "
                f"{unexpected_list}. Delete stale .state files or other unexpected "
                f"files in {self._root} so that {VOICE_MANIFEST_NAME} matches the "
                "stored .state files"
            )

    def _raise_if_managed_symlink_locked(self, path: Path) -> None:
        try:
            self._raise_if_symlink(path, "Voice store must not use symlinks")
        except RuntimeError:
            self._mark_store_tampered_locked("voice store must not use symlinks")

    def _raise_if_store_tampered_locked(self) -> None:
        if self._tamper_error is not None:
            raise self._tamper_error

    def _mark_store_tampered_locked(self, detail: str) -> None:
        if self._tamper_error is None:
            self._tamper_error = VoiceStoreTamperedError(
                "custom voice store is tampered; "
                f"custom voice functionality is disabled: {detail}"
            )
        raise self._tamper_error


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
    source_path = validate_source_audio_path(Path(path))
    source_fd = open_validated_source_audio_file(source_path)
    return await asyncio.to_thread(_copy_source_audio_sync, source_fd, spool_dir)


async def spool_voice_state_bytes(
    state_bytes: bytes,
    *,
    spool_dir: Path,
) -> Path:
    """Write one serialized custom-voice state into session-owned temp storage."""
    fd, temp_path = _create_owned_temp_file(spool_dir, suffix=".state")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(state_bytes)
        return temp_path
    except BaseException:
        unlink_if_exists(temp_path)
        raise


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
