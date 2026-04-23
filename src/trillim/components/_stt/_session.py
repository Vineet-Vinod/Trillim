from __future__ import annotations

import abc
import audioop
import io
import wave
from enum import Enum
from typing import TYPE_CHECKING

from trillim.components._stt._limits import MAX_UPLOAD_BYTES
from trillim.errors import InvalidRequestError, SessionBusyError

if TYPE_CHECKING:
    from trillim.components._stt.public import STT

_AUDIO_SESSION_OWNER_TOKEN = object()
_AUDIO_SESSION_CONSTRUCTION_ERROR = (
    "AudioSession cannot be constructed directly; use STT.open_session()"
)
_ALLOW_AUDIO_SESSION_SUBCLASS = False
_PCM_WIDTH_BYTES = 2
_PCM_CHANNELS = 1
_PCM_SAMPLE_RATE = 16000


class _AudioSessionFSM(Enum):
    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    DONE = "done"


def _create_audio_session(stt: STT) -> _AudioSession:
    return _AudioSession(stt, _owner_token=_AUDIO_SESSION_OWNER_TOKEN)


class AudioSession(abc.ABC):
    """Public audio-session handle returned by the STT component."""

    _runtime_proxy = True

    def __init_subclass__(cls, **kwargs) -> None:
        del kwargs
        super().__init_subclass__()
        if not _ALLOW_AUDIO_SESSION_SUBCLASS:
            raise TypeError("AudioSession cannot be subclassed publicly")

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        if cls is AudioSession:
            raise TypeError(_AUDIO_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    @property
    @abc.abstractmethod
    def state(self) -> str:
        ...

    @abc.abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        ...


_ALLOW_AUDIO_SESSION_SUBCLASS = True


class _AudioSession(AudioSession):
    """Private concrete audio-session implementation owned by one STT runtime."""

    def __new__(cls, stt: STT, *, _owner_token=None):
        del stt
        if _owner_token is not _AUDIO_SESSION_OWNER_TOKEN:
            raise TypeError(_AUDIO_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(self, stt: STT, *, _owner_token=None) -> None:
        if _owner_token is not _AUDIO_SESSION_OWNER_TOKEN:
            raise TypeError(_AUDIO_SESSION_CONSTRUCTION_ERROR)
        self._stt = stt
        self._state = _AudioSessionFSM.IDLE

    @property
    def state(self) -> str:
        return self._state.value

    async def transcribe(self, audio: bytes) -> str:
        if self._state is _AudioSessionFSM.TRANSCRIBING:
            raise SessionBusyError("AudioSession is already transcribing")

        self._state = _AudioSessionFSM.TRANSCRIBING
        try:
            if self._stopped():
                return ""
            pcm = self._normalize_audio(audio)
            if self._stopped():
                return ""
            return await self._stt._engine.transcribe(pcm)
        finally:
            self._state = _AudioSessionFSM.DONE

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
        if width == 1:
            pcm = audioop.bias(pcm, 1, -128)
        if width != _PCM_WIDTH_BYTES:
            pcm = audioop.lin2lin(pcm, width, _PCM_WIDTH_BYTES)
        if channels != _PCM_CHANNELS:
            pcm = audioop.tomono(pcm, _PCM_WIDTH_BYTES, 0.5, 0.5)
        if rate != _PCM_SAMPLE_RATE:
            pcm, _state = audioop.ratecv(
                pcm,
                _PCM_WIDTH_BYTES,
                _PCM_CHANNELS,
                rate,
                _PCM_SAMPLE_RATE,
                None,
            )
        return self._validate_pcm(pcm)

    def _is_wav(self, audio: bytes) -> bool:
        return len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"

    def _stopped(self) -> bool:
        stop_event = getattr(self._stt, "_stop_event", None)
        return bool(stop_event is not None and stop_event.is_set())


_ALLOW_AUDIO_SESSION_SUBCLASS = False
