from __future__ import annotations

import abc
import io
import wave
from enum import Enum
from typing import TYPE_CHECKING

from trillim.components.stt._limits import MAX_UPLOAD_BYTES
from trillim.errors import InvalidRequestError, SessionBusyError

if TYPE_CHECKING:
    from trillim.components.stt.public import STT

_STT_SESSION_OWNER_TOKEN = object()
_STT_SESSION_CONSTRUCTION_ERROR = (
    "STTSession cannot be constructed directly; use STT.open_session()"
)
_ALLOW_STT_SESSION_SUBCLASS = False
_PCM_WIDTH_BYTES = 2
_PCM_CHANNELS = 1
_PCM_SAMPLE_RATE = 16000


class _STTSessionFSM(Enum):
    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    DONE = "done"


def _create_stt_session(stt: STT) -> _STTSession:
    return _STTSession(stt, _owner_token=_STT_SESSION_OWNER_TOKEN)


class STTSession(abc.ABC):
    """Public STT session handle returned by the STT component."""

    _runtime_proxy = True

    def __init_subclass__(cls, **kwargs) -> None:
        del kwargs
        super().__init_subclass__()
        if not _ALLOW_STT_SESSION_SUBCLASS:
            raise TypeError("STTSession cannot be subclassed publicly")

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        if cls is STTSession:
            raise TypeError(_STT_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    @property
    @abc.abstractmethod
    def state(self) -> str:
        ...

    @abc.abstractmethod
    async def __aenter__(self) -> STTSession:
        ...

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        ...

    @abc.abstractmethod
    async def transcribe(self, audio: bytes, *, language: str | None = None) -> str:
        ...


_ALLOW_STT_SESSION_SUBCLASS = True


class _STTSession(STTSession):
    """Private concrete STT session implementation owned by one STT runtime."""

    def __new__(cls, stt: STT, *, _owner_token=None):
        del stt
        if _owner_token is not _STT_SESSION_OWNER_TOKEN:
            raise TypeError(_STT_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(self, stt: STT, *, _owner_token=None) -> None:
        if _owner_token is not _STT_SESSION_OWNER_TOKEN:
            raise TypeError(_STT_SESSION_CONSTRUCTION_ERROR)
        self._stt = stt
        self._state = _STTSessionFSM.IDLE

    @property
    def state(self) -> str:
        return self._state.value

    async def __aenter__(self) -> STTSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """No-op for now since we don't have streaming behavior"""
        return None

    async def transcribe(self, audio: bytes, *, language: str | None = None) -> str:
        if self._state is _STTSessionFSM.TRANSCRIBING:
            raise SessionBusyError("STTSession is already transcribing")

        self._state = _STTSessionFSM.TRANSCRIBING
        try:
            if self._stopped():
                return ""
            pcm = self._normalize_audio(audio)
            if self._stopped():
                return ""
            return await self._stt._transcribe(pcm, language=language)
        finally:
            self._state = _STTSessionFSM.DONE

    def _normalize_audio(self, audio: bytes) -> bytes:
        if isinstance(audio, bytearray):
            audio = bytes(audio)
        elif isinstance(audio, memoryview):
            audio = audio.tobytes()
        elif not isinstance(audio, bytes):
            raise InvalidRequestError("audio must be bytes")
        if not audio:
            raise InvalidRequestError("audio must not be empty")
        if self._is_wav(audio):
            return self._wav_to_pcm(audio)
        return self._validate_pcm(audio)

    def _validate_pcm(self, pcm: bytes) -> bytes:
        if len(pcm) % _PCM_WIDTH_BYTES != 0:
            raise InvalidRequestError("PCM audio must contain whole 16-bit samples")
        if len(pcm) > MAX_UPLOAD_BYTES:
            raise InvalidRequestError(f"PCM audio exceeds {MAX_UPLOAD_BYTES} bytes")
        return pcm

    def _wav_to_pcm(self, audio: bytes) -> bytes:
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav_file:
                if wav_file.getcomptype() != "NONE":
                    raise InvalidRequestError("WAV audio must be uncompressed PCM")
                pcm = wav_file.readframes(wav_file.getnframes())
                width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                rate = wav_file.getframerate()
        except (wave.Error, EOFError) as exc:
            raise InvalidRequestError("invalid WAV audio") from exc

        if not pcm:
            raise InvalidRequestError("audio must not be empty")
        if width <= 0 or channels <= 0 or rate <= 0:
            raise InvalidRequestError("invalid WAV audio")
        return self._validate_pcm(self._convert_wav_pcm(pcm, width, channels, rate))

    def _convert_wav_pcm(
        self,
        pcm: bytes,
        width: int,
        channels: int,
        rate: int,
    ) -> bytes:
        import numpy as np

        samples = self._decode_pcm_samples(pcm, width)
        if channels != _PCM_CHANNELS:
            frame_count = len(samples) // channels
            if frame_count * channels != len(samples):
                raise InvalidRequestError("invalid WAV audio")
            samples = samples.reshape(frame_count, channels).mean(axis=1)
        if rate != _PCM_SAMPLE_RATE:
            target_count = max(1, round(len(samples) * _PCM_SAMPLE_RATE / rate))
            source_positions = np.arange(len(samples), dtype=np.float32)
            target_positions = np.linspace(
                0.0,
                len(samples) - 1,
                num=target_count,
                dtype=np.float32,
            )
            samples = np.interp(target_positions, source_positions, samples)
        return self._encode_pcm_samples(samples)

    def _decode_pcm_samples(self, pcm: bytes, width: int):
        import numpy as np

        if width == 1:
            return (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if width == 2:
            return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if width == 4:
            return np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
        raise InvalidRequestError("unsupported WAV sample width")

    def _encode_pcm_samples(self, samples) -> bytes:
        import numpy as np

        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2").tobytes()

    def _is_wav(self, audio: bytes) -> bool:
        return len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"

    def _stopped(self) -> bool:
        stop_event = getattr(self._stt, "_stop_event", None)
        return bool(stop_event is not None and stop_event.is_set())


_ALLOW_STT_SESSION_SUBCLASS = False
