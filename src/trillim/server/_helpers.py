# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Shared utility functions for the Trillim SDK."""

import time
import uuid


def make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def now() -> int:
    return int(time.time())


from trillim.inference import load_default_params  # re-export
