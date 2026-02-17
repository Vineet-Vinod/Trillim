# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Shared utility functions for the Trillim SDK."""

import time
import uuid


def make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def now() -> int:
    return int(time.time())


def load_default_params(model_dir: str) -> dict:
    """Return hardcoded default sampling params."""
    return {
        "temperature": 0.6,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "rep_penalty_lookback": 64,
    }
