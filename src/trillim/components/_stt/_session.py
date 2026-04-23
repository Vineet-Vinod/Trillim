from __future__ import annotations

import abc
from asyncio import Lock, Event, AbstractEventLoop

from trillim.components._stt.public import STT

_AUDIO_SESSION_OWNER_TOKEN = object()
_AUDIO_SESSION_CONSTRUCTION_ERROR = (
    "AudioSession cannot be constructed directly; use STT.open_session()"
)
_ALLOW_AUDIO_SESSION_SUBCLASS = False


class _AudioSessionFSM()

def _create_audio_session(stt: STT) -> _AudioSession:
    return _AudioSession(stt, owner_token=_AUDIO_SESSION_OWNER_TOKEN)


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


_ALLOW_AUDIO_SESSION_SUBCLASS = True


class _AudioSession(AudioSession):
    """Private concrete audio-session implementation owned by one STT runtime."""

    def __new__(cls, stt: STT, *, _owner_token=None):
        del stt, lock, event
        if _owner_token is not _AUDIO_SESSION_OWNER_TOKEN:
            raise TypeError(_AUDIO_SESSION_CONSTRUCTION_ERROR)
        return super().__new__(cls)

    def __init__(self, stt: STT, *, _owner_token=None) -> None:
        if _owner_token is not _AUDIO_SESSION_OWNER_TOKEN:
            raise TypeError(_AUDIO_SESSION_CONSTRUCTION_ERROR)

        self._stt = stt


_ALLOW_AUDIO_SESSION_SUBCLASS = False