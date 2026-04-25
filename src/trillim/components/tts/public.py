"""Public TTS component and session API."""

from __future__ import annotations

import asyncio
import importlib
import tempfile
from asyncio import AbstractEventLoop
from pathlib import Path

from fastapi import APIRouter

from trillim.components import Component
from trillim.components.tts._engine import (
    TTSEngine,
    TTSEngineCrashedError,
    is_voice_cloning_auth_error,
)
from trillim.components.tts._limits import (
    DEFAULT_SPEED,
    VOICE_STORE_ROOT,
)
from trillim.components.tts._router import build_router
from trillim.components.tts._segmenter import load_pocket_tts_tokenizer
from trillim.components.tts._session import TTSSession, _create_tts_session
from trillim.components.tts._validation import (
    normalize_optional_name,
    normalize_required_name,
    validate_speed,
)
from trillim.components.tts._voices import (
    copy_source_audio,
    delete_custom_voice,
    load_custom_voice_states,
    publish_custom_voice,
    spool_voice_bytes,
)
from trillim.errors import ComponentLifecycleError, InvalidRequestError


DEFAULT_SESSION_VOICE = "alba"
_CLIENT_VOICE_BUILD_ERROR_SNIPPETS = (
    "unsupported or malformed audio input",
    "unsupported audio input",
    "malformed audio input",
    "format not recognised",
    "unknown format",
    "invalid data found when processing input",
    "custom voice state exceeds the",
)


class TTS(Component):
    """PocketTTS-backed text-to-speech component."""

    def __init__(self) -> None:
        self._engine = TTSEngine()
        self._lifecycle_lock = asyncio.Lock()
        self._synthesize_lock = asyncio.Lock()
        self._tokenizer_lock = asyncio.Lock()
        self._tokenizer = None
        self._owner_loop: AbstractEventLoop | None = None
        self._built_in_voice_names: tuple[str, ...] = ()
        self._voice_state_cache: dict[str, str | dict] = {}
        self._spool_dir = Path(tempfile.gettempdir()) / "trillim-tts"
        self._stop_event = asyncio.Event()
        self._stop_event.set()
        self._started = False

    def router(self) -> APIRouter:
        """Return the TTS HTTP router."""
        return build_router(self)

    async def start(self) -> None:
        """Start PocketTTS and initialize voice storage."""
        async with self._lifecycle_lock:
            self._require_owner_loop()
            if self._started:
                return
            importlib.import_module("numpy")
            importlib.import_module("soundfile")
            importlib.import_module("pocket_tts")
            built_in_voice_names = _load_built_in_voice_names()
            await self._engine.start()
            custom_voice_states = await load_custom_voice_states(
                VOICE_STORE_ROOT,
                built_in_voice_names=built_in_voice_names,
            )
            self._built_in_voice_names = built_in_voice_names
            self._voice_state_cache = {
                **{name: name for name in built_in_voice_names},
                **custom_voice_states,
            }
            self._stop_event.clear()
            self._started = True

    async def stop(self) -> None:
        """Stop PocketTTS and clear in-memory component state."""
        self._require_owner_loop()
        async with self._lifecycle_lock:
            if not self._started:
                return
            self._started = False
            self._stop_event.set()
            self._voice_state_cache.clear()
            async with self._tokenizer_lock:
                self._tokenizer = None
            async with self._synthesize_lock:
                await self._engine.stop()

    async def list_voices(self) -> list[str]:
        """Return the visible built-in and custom voice names."""
        self._require_owner_loop()
        self._require_started()
        custom_names = sorted(
            name for name in self._voice_state_cache if name not in self._built_in_voice_names
        )
        return [*self._built_in_voice_names, *custom_names]

    async def register_voice(self, name: str, audio: bytes | str | Path) -> str:
        """Register one custom voice from bytes or one caller-owned path."""
        self._require_owner_loop()
        self._require_started()
        normalized_name = normalize_required_name(name, field_name="name")
        if normalized_name in self._voice_state_cache:
            raise InvalidRequestError(f"voice '{normalized_name}' already exists")
        if isinstance(audio, bytes):
            owned_upload = await spool_voice_bytes(audio, spool_dir=self._spool_dir)
        elif isinstance(audio, (str, Path)):
            owned_upload = await copy_source_audio(audio, spool_dir=self._spool_dir)
        else:
            raise InvalidRequestError("audio must be bytes, str, or Path")
        try:
            voice_state = await self._build_voice_state(owned_upload.path)
            registered_name, voice_state = await publish_custom_voice(
                VOICE_STORE_ROOT,
                name=normalized_name,
                voice_state=voice_state,
                existing_names=set(self._voice_state_cache) - set(self._built_in_voice_names),
            )
            self._voice_state_cache[registered_name] = voice_state
            return registered_name
        finally:
            owned_upload.path.unlink(missing_ok=True)

    async def delete_voice(self, name: str) -> str:
        """Delete one managed custom voice."""
        self._require_owner_loop()
        self._require_started()
        normalized_name = normalize_required_name(name, field_name="name")
        if normalized_name in self._built_in_voice_names:
            raise InvalidRequestError(
                f"voice '{normalized_name}' is built in and cannot be deleted"
            )
        if normalized_name not in self._voice_state_cache:
            raise KeyError(normalized_name)
        self._voice_state_cache.pop(normalized_name, None)
        deleted_name = await delete_custom_voice(VOICE_STORE_ROOT, name=normalized_name)
        return deleted_name

    async def open_session(
        self,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> TTSSession:
        """Open one reusable TTS session."""
        self._require_owner_loop()
        self._require_started()
        resolved_voice = (
            DEFAULT_SESSION_VOICE
            if voice is None
            else normalize_optional_name(voice, field_name="voice")
        )
        assert resolved_voice is not None
        resolved_voice, _voice_state = await self._configure_voice(resolved_voice)
        return _create_tts_session(
            self,
            voice=resolved_voice,
            speed=DEFAULT_SPEED if speed is None else validate_speed(speed),
        )

    async def _configure_voice(self, voice: str) -> tuple[str, str | dict]:
        self._require_owner_loop()
        self._require_started()
        normalized = normalize_required_name(voice, field_name="voice")
        cached = self._voice_state_cache.get(normalized)
        if cached is None:
            raise InvalidRequestError(f"unknown voice: {normalized}")
        return normalized, cached

    async def _synthesize_segment(
        self,
        text: str,
        voice_state: object,
    ) -> bytes:
        self._require_owner_loop()
        self._require_started()
        if not isinstance(voice_state, (str, dict)):
            raise InvalidRequestError("voice state is malformed")
        async with self._synthesize_lock:
            self._require_started()
            return await self._engine.synthesize_segment(
                text,
                voice_state=voice_state,
            )

    async def _get_tokenizer(self):
        self._require_owner_loop()
        async with self._tokenizer_lock:
            self._require_started()
            if self._tokenizer is None:
                self._tokenizer = load_pocket_tts_tokenizer()
            return self._tokenizer

    async def _build_voice_state(self, audio_path: Path) -> dict:
        try:
            async with self._synthesize_lock:
                self._require_started()
                return await self._engine.build_voice_state(audio_path)
        except TTSEngineCrashedError as exc:
            message = str(exc)
            if _is_client_voice_build_error(message):
                raise InvalidRequestError(message) from exc
            raise

    def _require_started(self) -> None:
        if self._stopped():
            raise ComponentLifecycleError("TTS is not running")

    def _require_owner_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
            return
        if loop is not self._owner_loop:
            raise ComponentLifecycleError(
                "TTS is bound to one event loop; create a new TTS per thread/event loop"
            )

    def _stopped(self) -> bool:
        return not self._started or self._stop_event.is_set()


def _load_built_in_voice_names() -> tuple[str, ...]:
    from pocket_tts.utils.utils import PREDEFINED_VOICES

    return tuple(PREDEFINED_VOICES)


def _is_client_voice_build_error(message: str) -> bool:
    text = message.lower()
    return is_voice_cloning_auth_error(message) or any(
        snippet in text for snippet in _CLIENT_VOICE_BUILD_ERROR_SNIPPETS
    )
