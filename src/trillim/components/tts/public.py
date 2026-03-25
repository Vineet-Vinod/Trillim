"""Public TTS component and session API."""

from __future__ import annotations

import asyncio
import importlib
import struct
import tempfile
from collections import deque
from pathlib import Path

from fastapi import APIRouter

from trillim.components import Component
from trillim.components.tts._limits import (
    DEFAULT_SPEED,
    PCM_CHANNELS,
    PCM_SAMPLE_RATE,
    PCM_SAMPLE_WIDTH_BYTES,
    VOICE_STORE_ROOT,
)
from trillim.components.tts._router import build_router
from trillim.components.tts._segmenter import iter_text_segments, load_pocket_tts_tokenizer
from trillim.components.tts._session import TTSSession, _TTSSession, _create_tts_session
from trillim.components.tts._validation import (
    normalize_optional_name,
    normalize_required_name,
    validate_speed,
    validate_text,
)
from trillim.components.tts._voices import (
    VoiceStore,
    copy_source_audio,
    spool_request_voice_stream,
    spool_voice_bytes,
)
from trillim.components.tts._worker import (
    WorkerFailureError,
    build_voice_state,
    create_session_worker,
    is_voice_cloning_auth_error,
)
from trillim.errors import AdmissionRejectedError, InvalidRequestError, SessionClosedError

DEFAULT_VOICE = "alba"
_CLIENT_VOICE_BUILD_ERROR_SNIPPETS = (
    "unsupported or malformed audio input",
    "unsupported audio input",
    "malformed audio input",
    "format not recognised",
    "format not recognized",
    "unknown format",
    "invalid data found when processing input",
)


class TTS(Component):
    """PocketTTS-backed text-to-speech component."""

    def __init__(
        self,
        *,
        default_voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        _tokenizer_loader=load_pocket_tts_tokenizer,
        _session_worker_factory=create_session_worker,
        _voice_state_builder=build_voice_state,
    ) -> None:
        self._default_voice = normalize_required_name(
            default_voice,
            field_name="default_voice",
        )
        self._default_speed = validate_speed(speed)
        self._tokenizer_loader = _tokenizer_loader
        self._session_worker_factory = _session_worker_factory
        self._voice_state_builder = _voice_state_builder
        self._started = False
        self._accepting = False
        self._scheduler_lock = asyncio.Lock()
        self._tokenizer_lock = asyncio.Lock()
        self._tokenizer = None
        self._voice_store: VoiceStore | None = None
        self._spool_dir = Path(tempfile.gettempdir()) / "trillim-tts"
        self._active_session: _TTSSession | None = None
        self._reserved_slot: object | None = None

    @property
    def default_voice(self) -> str:
        """Return the configured default voice name."""
        return self._default_voice

    @property
    def speed(self) -> float:
        """Return the configured default synthesis speed."""
        return self._default_speed

    def router(self) -> APIRouter:
        """Return the TTS HTTP router."""
        return build_router(self)

    async def start(self) -> None:
        """Verify optional voice dependencies and initialize the voice store."""
        if self._started:
            return
        importlib.import_module("numpy")
        importlib.import_module("soundfile")
        importlib.import_module("pocket_tts")
        built_in_voice_names = _load_built_in_voice_names()
        voice_store = VoiceStore(
            VOICE_STORE_ROOT,
            built_in_voice_names=built_in_voice_names,
        )
        if self._default_voice not in built_in_voice_names:
            available = await voice_store.list_names()
            if self._default_voice not in available:
                raise ValueError(f"unknown default_voice: {self._default_voice}")
        self._voice_store = voice_store
        self._started = True
        self._accepting = True

    async def stop(self) -> None:
        """Drain admissions and stop all live TTS sessions."""
        active_session: _TTSSession | None = None
        active_task: asyncio.Task | None = None
        async with self._scheduler_lock:
            self._accepting = False
            self._reserved_slot = None
            active_session = self._active_session
            self._active_session = None
            if active_session is not None:
                active_session._mark_owner_stopped()
                active_task = active_session._task
                if active_task is not None:
                    active_task.cancel()
        if active_session is not None:
            if active_task is not None:
                await asyncio.gather(active_task, return_exceptions=True)
            if not active_session._done_event.is_set():
                await active_session._finish(
                    "owner_stopped",
                    SessionClosedError("TTSSession owner has stopped"),
                )
            await active_session._wait_for_done()
        self._started = False

    async def list_voices(self) -> list[str]:
        """Return the visible built-in and custom voice names."""
        return await self._require_voice_store().list_names()

    async def register_voice(self, name: str, audio: bytes | str | Path) -> str:
        """Register one custom voice from bytes or one caller-owned path."""
        voice_store = self._require_voice_store()
        normalized_name = normalize_required_name(name, field_name="name")
        await voice_store.ensure_name_available(normalized_name)
        if isinstance(audio, bytes):
            owned_upload = await spool_voice_bytes(audio, spool_dir=self._spool_dir)
        elif isinstance(audio, (str, Path)):
            owned_upload = await copy_source_audio(audio, spool_dir=self._spool_dir)
        else:
            raise InvalidRequestError("audio must be bytes, str, or Path")
        try:
            return await voice_store.register_owned_upload(
                name=normalized_name,
                upload=owned_upload,
                build_voice_state=self._build_voice_state,
            )
        finally:
            owned_upload.path.unlink(missing_ok=True)

    async def delete_voice(self, name: str) -> str:
        """Delete one managed custom voice."""
        normalized_name = normalize_required_name(name, field_name="name")
        return await self._require_voice_store().delete(
            normalized_name,
            protected_name=self._default_voice,
        )

    async def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> TTSSession:
        """Create and schedule one live TTS session."""
        self._require_started()
        normalized_text = validate_text(text)
        resolved_voice_name = (
            self._default_voice
            if voice is None
            else normalize_optional_name(voice, field_name="voice")
        )
        assert resolved_voice_name is not None
        resolved_speed = self._default_speed if speed is None else validate_speed(speed)
        reservation = await self._reserve_session_slot()
        try:
            session = await self._start_reserved_session(
                reservation,
                normalized_text,
                voice=resolved_voice_name,
                speed=resolved_speed,
            )
            reservation = None
            return session
        finally:
            if reservation is not None:
                await self._release_reserved_slot(reservation)

    async def synthesize_wav(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        """Synthesize one whole WAV payload through the session pipeline."""
        async with await self.speak(text, voice=voice, speed=speed) as session:
            pcm = await session.collect()
        return _wav_header(data_size=len(pcm)) + pcm

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ):
        """Stream raw PCM bytes through the session pipeline."""
        async with await self.speak(text, voice=voice, speed=speed) as session:
            async for chunk in session:
                yield chunk

    async def _pause_session(self, session: _TTSSession) -> None:
        async with self._scheduler_lock:
            if session._done_event.is_set():
                return
            if session is not self._active_session:
                return
            if session.state == "paused":
                return
            if session._chunk_in_flight:
                session._pause_requested = True
            else:
                self._pause_active_locked(session)

    async def _resume_session(self, session: _TTSSession) -> None:
        async with self._scheduler_lock:
            if session._done_event.is_set():
                return
            if session is not self._active_session or session.state != "paused":
                return
            session._pause_requested = False
            session._set_running()

    async def _cancel_session(self, session: _TTSSession) -> None:
        active_task: asyncio.Task | None = None
        async with self._scheduler_lock:
            if session._done_event.is_set():
                return
            if session is self._active_session:
                session._pause_requested = False
                active_task = session._task
                if active_task is not None:
                    active_task.cancel()
        if active_task is not None:
            await asyncio.gather(active_task, return_exceptions=True)
            if not session._done_event.is_set():
                await session._finish("cancelled", session._cancel_error())
        await session._wait_for_done()

    async def _set_session_speed(self, session: _TTSSession, speed: float) -> None:
        async with self._scheduler_lock:
            if session._done_event.is_set():
                return
            session._speed = speed

    async def _reserve_session_slot(self) -> object:
        self._require_started()
        async with self._scheduler_lock:
            self._check_can_accept_locked()
            token = object()
            self._reserved_slot = token
            return token

    async def _release_reserved_slot(self, token: object) -> None:
        async with self._scheduler_lock:
            if self._reserved_slot is token:
                self._reserved_slot = None

    async def _start_reserved_session(
        self,
        token: object,
        text: str,
        *,
        voice: str,
        speed: float,
    ) -> TTSSession:
        self._require_started()
        resolved_voice = await self._require_voice_store().resolve_for_session(
            voice,
            spool_dir=self._spool_dir,
        )
        session_worker = self._session_worker_factory(
            voice_kind=resolved_voice.kind,
            voice_reference=resolved_voice.reference,
        )
        session = _create_tts_session(
            self,
            text=text,
            voice=voice,
            voice_kind=resolved_voice.kind,
            voice_reference=resolved_voice.reference,
            speed=speed,
            cleanup_path=resolved_voice.cleanup_path,
            session_worker=session_worker,
        )
        started = False
        try:
            async with self._scheduler_lock:
                if self._reserved_slot is not token:
                    self._check_can_accept_locked()
                    raise AdmissionRejectedError(
                        "TTS reservation is no longer active; request cannot start"
                    )
                self._reserved_slot = None
                self._start_session_locked(session)
                started = True
            return session
        finally:
            if not started:
                if resolved_voice.cleanup_path is not None:
                    resolved_voice.cleanup_path.unlink(missing_ok=True)
                await session_worker.close()

    async def _register_voice_http_request(self, request) -> str:
        """Handle the raw-body HTTP voice-upload request."""
        voice_store = self._require_voice_store()
        metadata = request.state.trillim_tts_voice_request
        await voice_store.ensure_name_available(metadata.name)
        owned_upload = await spool_request_voice_stream(
            request.stream(),
            spool_dir=self._spool_dir,
        )
        try:
            return await voice_store.register_owned_upload(
                name=metadata.name,
                upload=owned_upload,
                build_voice_state=self._build_voice_state,
            )
        finally:
            owned_upload.path.unlink(missing_ok=True)

    async def _build_voice_state(self, audio_path: Path) -> bytes:
        try:
            return await self._voice_state_builder(audio_path)
        except WorkerFailureError as exc:
            message = str(exc)
            if _is_client_voice_build_error(message):
                raise InvalidRequestError(message) from exc
            raise

    async def _run_session(self, session: _TTSSession) -> None:
        try:
            tokenizer = await self._get_tokenizer()
            for segment in iter_text_segments(session._text, tokenizer):
                speed = await self._wait_for_turn(session)
                pcm = await session._session_worker.synthesize(segment)
                stretched = _stretch_pcm_chunk(pcm, speed)
                await session._put_chunk(stretched)
                async with self._scheduler_lock:
                    session._chunk_in_flight = False
                    if session._pause_requested and session is self._active_session:
                        self._pause_active_locked(session)
            await self._complete_running_session(session)
        except asyncio.CancelledError:
            state = "owner_stopped" if session.state == "owner_stopped" else "cancelled"
            error = (
                SessionClosedError("TTSSession owner has stopped")
                if state == "owner_stopped"
                else session._cancel_error()
            )
            await self._finalize_running_session(
                session,
                state=state,
                error=error,
            )
        except Exception as exc:
            await self._finalize_running_session(
                session,
                state="failed",
                error=exc,
            )

    async def _wait_for_turn(self, session: _TTSSession) -> float:
        while True:
            await session._resume_event.wait()
            async with self._scheduler_lock:
                if session._done_event.is_set():
                    raise session._cancel_error()
                if session is not self._active_session:
                    continue
                if session._pause_requested:
                    self._pause_active_locked(session)
                    continue
                session._chunk_in_flight = True
                return session.speed

    async def _complete_running_session(self, session: _TTSSession) -> None:
        await self._finalize_running_session(session, state="completed", error=None)

    async def _finalize_running_session(
        self,
        session: _TTSSession,
        *,
        state: str,
        error: Exception | None,
    ) -> None:
        async with self._scheduler_lock:
            session._chunk_in_flight = False
        close_error: Exception | None = None
        try:
            await session._session_worker.close()
        except Exception as exc:
            close_error = exc
        effective_state = state
        effective_error = error
        if close_error is not None and error is None:
            effective_state = "failed"
            effective_error = close_error
        async with self._scheduler_lock:
            try:
                await session._finish(effective_state, effective_error)
            finally:
                if self._active_session is session:
                    self._active_session = None

    async def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        async with self._tokenizer_lock:
            if self._tokenizer is None:
                self._tokenizer = self._tokenizer_loader()
        return self._tokenizer

    def _start_session_locked(self, session: _TTSSession) -> None:
        self._active_session = session
        session._pause_requested = False
        session._set_running()
        if session._task is None:
            session._task = asyncio.create_task(self._run_session(session))

    def _pause_active_locked(self, session: _TTSSession) -> None:
        if self._active_session is not session:
            return
        session._pause_requested = False
        session._set_paused()

    def _check_can_accept_locked(self) -> None:
        if not self._accepting:
            raise AdmissionRejectedError("TTS is draining and not accepting new requests")
        if self._reserved_slot is not None:
            raise AdmissionRejectedError("TTS is busy; only one live session is allowed")
        if self._active_session is not None:
            raise AdmissionRejectedError("TTS is busy; only one live session is allowed")

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("TTS is not started")

    def _require_voice_store(self) -> VoiceStore:
        self._require_started()
        assert self._voice_store is not None
        return self._voice_store


def _load_built_in_voice_names() -> tuple[str, ...]:
    from pocket_tts.utils.utils import PREDEFINED_VOICES

    return tuple(PREDEFINED_VOICES)


def _is_client_voice_build_error(message: str) -> bool:
    text = message.lower()
    return is_voice_cloning_auth_error(message) or any(
        snippet in text for snippet in _CLIENT_VOICE_BUILD_ERROR_SNIPPETS
    )


def _wav_header(data_size: int) -> bytes:
    byte_rate = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
    block_align = PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        PCM_CHANNELS,
        PCM_SAMPLE_RATE,
        byte_rate,
        block_align,
        PCM_SAMPLE_WIDTH_BYTES * 8,
        b"data",
        data_size,
    )


def _stretch_pcm_chunk(pcm: bytes, speed: float) -> bytes:
    if speed == 1.0 or not pcm:
        return pcm
    stretcher = _StreamingPCMStretcher(speed)
    return stretcher.push(pcm) + stretcher.finish()


class _StreamingPCMStretcher:
    def __init__(self, speed: float):
        import numpy as np

        self._np = np
        self.speed = validate_speed(speed)
        self.frame_size = 1024
        self.hop_size = self.frame_size // 4
        self._window = np.hanning(self.frame_size).astype(np.float32)
        if not np.any(self._window):
            self._window = np.ones(self.frame_size, dtype=np.float32)
        self._window_sq = self._window**2
        self._phase_advance = (
            2.0
            * np.pi
            * self.hop_size
            * np.arange(self.frame_size // 2 + 1, dtype=np.float32)
            / self.frame_size
        )
        self._input = np.zeros(0, dtype=np.float32)
        self._input_base = 0
        self._total_samples = 0
        self._next_analysis_frame = 0
        self._spectra = deque()
        self._spectra_base = 0
        self._phase = None
        self._next_time_step = 0.0
        self._processed_output_frames = 0
        self._output = np.zeros(0, dtype=np.float32)
        self._weights = np.zeros(0, dtype=np.float32)
        self._output_base = 0
        self._pending_byte = b""

    def push(self, pcm_bytes: bytes) -> bytes:
        self._append_pcm(pcm_bytes)
        self._materialize_analysis_frames(final=False)
        self._process_output_frames(final=False)
        return self._emit_ready(final=False)

    def finish(self) -> bytes:
        self._materialize_analysis_frames(final=True)
        self._process_output_frames(final=True)
        return self._emit_ready(final=True)

    def _append_pcm(self, pcm_bytes: bytes) -> None:
        np = self._np
        data = self._pending_byte + pcm_bytes
        if len(data) % 2 == 1:
            self._pending_byte = data[-1:]
            data = data[:-1]
        else:
            self._pending_byte = b""
        if not data:
            return
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if self._input.size == 0:
            self._input = samples.copy()
        else:
            self._input = np.concatenate((self._input, samples))
        self._total_samples += samples.size

    def _final_frame_limit(self) -> int:
        if self._total_samples == 0:
            return 0
        return max(1, self._total_samples - self.frame_size) + self.hop_size

    def _materialize_analysis_frames(self, *, final: bool) -> None:
        np = self._np
        available_end = self._input_base + self._input.size
        final_limit = self._final_frame_limit()
        while True:
            start = self._next_analysis_frame * self.hop_size
            end = start + self.frame_size
            if end <= available_end:
                local_start = start - self._input_base
                frame = self._input[local_start : local_start + self.frame_size]
            elif final and start < final_limit:
                local_start = start - self._input_base
                frame = self._input[local_start:]
                if frame.size < self.frame_size:
                    frame = np.pad(frame, (0, self.frame_size - frame.size))
            else:
                break
            spectrum = np.fft.rfft(frame * self._window).astype(np.complex64)
            self._spectra.append(spectrum)
            self._next_analysis_frame += 1
        trim_to = self._next_analysis_frame * self.hop_size
        drop = max(0, min(self._input.size, trim_to - self._input_base))
        if drop > 0:
            self._input = self._input[drop:]
            self._input_base += drop

    def _analysis_frame_count(self) -> int:
        return self._spectra_base + len(self._spectra)

    def _get_spectrum(self, frame_index: int):
        local_index = frame_index - self._spectra_base
        if local_index < 0 or local_index >= len(self._spectra):
            return None
        return self._spectra[local_index]

    def _process_output_frames(self, *, final: bool) -> None:
        np = self._np
        frame_count = self._analysis_frame_count()
        while True:
            if self._phase is None:
                first = self._get_spectrum(0)
                if first is None:
                    return
                self._phase = np.angle(first).astype(np.float32)
                magnitude = np.abs(first)
                stretched = magnitude * np.exp(1j * self._phase)
                self._add_output_frame(
                    np.fft.irfft(stretched, n=self.frame_size).real.astype(np.float32)
                )
                self._next_time_step = float(self.speed)
                continue
            step = self._next_time_step
            current_index = int(step)
            if final:
                if step >= frame_count:
                    break
            elif current_index + 1 >= frame_count:
                break
            next_index = min(current_index + 1, frame_count - 1)
            current = self._get_spectrum(current_index)
            following = self._get_spectrum(next_index)
            if current is None or following is None:
                break
            fraction = step - current_index
            current_mag = np.abs(current)
            following_mag = np.abs(following)
            magnitude = current_mag * (1.0 - fraction) + following_mag * fraction
            delta = np.angle(following) - np.angle(current) - self._phase_advance
            delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
            phase_increment = self._phase_advance + delta
            self._phase = (self._phase + phase_increment).astype(np.float32)
            stretched = magnitude * np.exp(1j * self._phase)
            self._add_output_frame(
                np.fft.irfft(stretched, n=self.frame_size).real.astype(np.float32)
            )
            self._next_time_step += self.speed

    def _add_output_frame(self, frame) -> None:
        np = self._np
        output_start = self._output_base + (self._processed_output_frames * self.hop_size)
        output_end = output_start + self.frame_size
        needed = output_end - self._output_base
        if needed > self._output.size:
            pad = needed - self._output.size
            self._output = np.pad(self._output, (0, pad))
            self._weights = np.pad(self._weights, (0, pad))
        local_start = output_start - self._output_base
        self._output[local_start : local_start + self.frame_size] += frame * self._window
        self._weights[local_start : local_start + self.frame_size] += self._window_sq
        self._processed_output_frames += 1

    def _emit_ready(self, *, final: bool) -> bytes:
        np = self._np
        if self._processed_output_frames == 0:
            return b""
        ready_frames = self._processed_output_frames if final else max(0, self._processed_output_frames - 2)
        if ready_frames <= 0:
            return b""
        local_end = ready_frames * self.hop_size
        audio = self._output[:local_end]
        weights = self._weights[:local_end]
        nonzero = weights > 1e-8
        normalized = np.zeros_like(audio)
        normalized[nonzero] = audio[nonzero] / weights[nonzero]
        clipped = np.clip(normalized, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16).tobytes()
        self._output = self._output[local_end:]
        self._weights = self._weights[local_end:]
        self._output_base += local_end
        self._drop_consumed_spectra(ready_frames)
        self._processed_output_frames -= ready_frames
        return pcm

    def _drop_consumed_spectra(self, ready_frames: int) -> None:
        keep_from = max(0, ready_frames - 1)
        while self._spectra_base < keep_from:
            if not self._spectra:
                break
            self._spectra.popleft()
            self._spectra_base += 1
