# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Harness registry — maps names to harness classes."""

from ._base import Harness
from ._default import DefaultHarness

HARNESS_REGISTRY: dict[str, type[Harness]] = {
    "default": DefaultHarness,
}


def get_harness(name: str) -> type[Harness]:
    if name not in HARNESS_REGISTRY:
        available = ", ".join(sorted(HARNESS_REGISTRY))
        raise ValueError(f"Unknown harness {name!r}. Available: {available}")
    return HARNESS_REGISTRY[name]
