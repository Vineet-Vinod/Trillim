"""Fixed internal STT configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Fixed worker runtime settings."""

    model_name: str
    device: str
    compute_type: str


DEFAULT_WORKER_CONFIG = WorkerConfig(
    model_name="base",
    device="cpu",
    compute_type="int8",
)
