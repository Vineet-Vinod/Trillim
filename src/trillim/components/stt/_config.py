"""Fixed internal STT configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OwnedAudioInput:
    """A Trillim-owned normalized audio file."""

    path: Path
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    """Best-effort metadata snapshot for a caller-owned source file."""

    size_bytes: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Fixed worker runtime settings for Phase 4."""

    model_name: str
    device: str
    compute_type: str


DEFAULT_WORKER_CONFIG = WorkerConfig(
    model_name="base",
    device="cpu",
    compute_type="int8",
)
