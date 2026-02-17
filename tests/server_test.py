# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""
Test suite for the Trillim OpenAI-compatible API server.

Start the server first:  trillim serve <model_dir> --voice
Then run this:           uv run tests/server_test.py [--base-url URL] [--model-dir DIR] [--adapter-dir DIR]

Voice tests run automatically when the server has --voice enabled;
otherwise they are skipped.  LoRA tests require --adapter-dir pointing to a
directory with qmodel.lora (separate from --model-dir).
"""

import argparse
import json
import struct
import sys
import threading
import urllib.error
import urllib.request
import uuid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api(base_url: str, method: str, path: str, body=None, timeout: int = 300):
    """Make an API request and return (status_code, parsed_json | None)."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read() if exc.fp else b""
        try:
            return exc.code, json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            return exc.code, {"raw": body_bytes.decode(errors="replace")}


def stream_chunks(base_url: str, path: str, body: dict, timeout: int = 300):
    """Make a streaming request and return a list of parsed SSE data objects
    plus a boolean indicating whether the final ``data: [DONE]`` sentinel was
    received."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    chunks: list[dict] = []
    got_done = False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    got_done = True
                    break
                chunks.append(json.loads(payload))
    return chunks, got_done


def multipart(base_url: str, path: str, fields: dict[str, str],
              files: dict[str, tuple[str, bytes, str]],
              timeout: int = 300) -> tuple[int, bytes, dict[str, str]]:
    """POST a multipart/form-data request.

    *fields*: name -> value  (text form fields)
    *files*:  name -> (filename, data, content_type)

    Returns (status_code, response_body_bytes, response_headers).
    """
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode()
        )

    for name, (filename, data, ctype) in files.items():
        parts.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )

    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    url = f"{base_url}{path}"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, resp.read(), headers
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read() if exc.fp else b""
        headers = {k.lower(): v for k, v in exc.headers.items()}
        return exc.code, body_bytes, headers


def api_binary(base_url: str, method: str, path: str, body=None,
               timeout: int = 300) -> tuple[int, bytes, dict[str, str]]:
    """Make an API request and return (status, raw_bytes, headers)."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, resp.read(), headers
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read() if exc.fp else b""
        headers = {k.lower(): v for k, v in exc.headers.items()}
        return exc.code, body_bytes, headers


def _voice_enabled(base_url: str) -> bool:
    """Return True if the server has the voice pipeline enabled."""
    try:
        status, _, _ = api_binary(base_url, "GET", "/v1/voices", timeout=5)
        return status == 200
    except Exception:
        return False


def _is_valid_wav(data: bytes) -> bool:
    """Check that *data* starts with a valid RIFF/WAVE header."""
    if len(data) < 44:
        return False
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _wav_sample_rate(data: bytes) -> int:
    """Extract sample rate from a WAV header."""
    return struct.unpack_from("<I", data, 24)[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_models_endpoint(base_url: str, **_):
    """GET /v1/models returns a list with at least one model."""
    status, body = api(base_url, "GET", "/v1/models")
    if status != 200:
        return "fail", f"expected 200, got {status}"
    if "data" not in body:
        return "fail", "response missing 'data'"
    if len(body["data"]) < 1:
        return "fail", "no models returned"
    if body["data"][0]["object"] != "model":
        return "fail", "first entry not a model object"


def test_chat_non_streaming(base_url: str, **_):
    """POST /v1/chat/completions (non-streaming) returns a valid response."""
    payload = {
        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        "max_tokens": 64,
    }
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str) or len(content) == 0:
        return "fail", "empty response content"
    usage = body.get("usage", {})
    if usage.get("prompt_tokens", 0) <= 0:
        return "fail", "prompt_tokens should be > 0"
    if usage.get("completion_tokens", 0) <= 0:
        return "fail", "completion_tokens should be > 0"


def test_chat_streaming(base_url: str, **_):
    """POST /v1/chat/completions (streaming) yields correct SSE chunks."""
    payload = {
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 32,
        "stream": True,
    }
    chunks, got_done = stream_chunks(base_url, "/v1/chat/completions", payload)
    if len(chunks) < 2:
        return "fail", f"expected >=2 chunks, got {len(chunks)}"

    first_delta = chunks[0]["choices"][0].get("delta", {})
    if first_delta.get("role") != "assistant":
        return "fail", "first chunk missing role=assistant"

    content_pieces = [
        c["choices"][0]["delta"].get("content", "")
        for c in chunks[1:-1]
    ]
    if not any(content_pieces):
        return "fail", "no content in middle chunks"

    last_choice = chunks[-1]["choices"][0]
    if last_choice.get("finish_reason") != "stop":
        return "fail", "last chunk missing finish_reason=stop"

    if not got_done:
        return "fail", "stream did not end with data: [DONE]"


def test_completions_non_streaming(base_url: str, **_):
    """POST /v1/completions (non-streaming) returns text."""
    payload = {
        "prompt": "Once upon a time",
        "max_tokens": 32,
    }
    status, body = api(base_url, "POST", "/v1/completions", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    text = body["choices"][0]["text"]
    if not isinstance(text, str) or len(text) == 0:
        return "fail", "empty completion text"


def test_completions_streaming(base_url: str, **_):
    """POST /v1/completions (streaming) yields SSE chunks."""
    payload = {
        "prompt": "The sky is",
        "max_tokens": 32,
        "stream": True,
    }
    chunks, got_done = stream_chunks(base_url, "/v1/completions", payload)
    if len(chunks) < 1:
        return "fail", "no chunks received"
    if not got_done:
        return "fail", "stream did not end with data: [DONE]"

    text_pieces = [c["choices"][0].get("text", "") for c in chunks[:-1]]
    if not any(text_pieces):
        return "fail", "no text in streaming completion chunks"


def test_greedy_decoding(base_url: str, **_):
    """temperature=0 produces deterministic output across two runs."""
    payload = {
        "messages": [{"role": "user", "content": "What is the capital of the United States?"}],
        "temperature": 0,
        "max_tokens": 32,
    }
    s1, body1 = api(base_url, "POST", "/v1/chat/completions", payload)
    if s1 != 200:
        return "fail", f"first request failed: {s1}"
    s2, body2 = api(base_url, "POST", "/v1/chat/completions", payload)
    if s2 != 200:
        return "fail", f"second request failed: {s2}"
    r1 = body1["choices"][0]["message"]["content"]
    r2 = body2["choices"][0]["message"]["content"]
    if r1 != r2:
        return "fail", f"greedy decoding not deterministic:\n  run1: {r1!r}\n  run2: {r2!r}"


def test_max_tokens(base_url: str, **_):
    """max_tokens is respected."""
    payload = {
        "messages": [{"role": "user", "content": "Tell me a long story."}],
        "max_tokens": 5,
    }
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    completion_tokens = body.get("usage", {}).get("completion_tokens", 0)
    if completion_tokens > 5:
        return "fail", f"completion_tokens={completion_tokens}, expected <= 5"


def test_cached_tokens_field(base_url: str, **_):
    """Non-streaming responses include cached_tokens in usage info."""
    payload = {
        "messages": [{"role": "user", "content": "What color is the sky?"}],
        "max_tokens": 16,
    }
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    usage = body.get("usage", {})
    if "cached_tokens" not in usage:
        return "fail", "usage missing 'cached_tokens' field"
    if not isinstance(usage["cached_tokens"], int):
        return "fail", f"cached_tokens should be int, got {type(usage['cached_tokens']).__name__}"


def test_cache_hit_multi_turn(base_url: str, **_):
    """Second request with an extended message list gets a cache hit."""
    # First request — establishes the cache (use temperature=0 for determinism)
    msg1 = [{"role": "user", "content": "Name a primary color. One word."}]
    payload1 = {"messages": msg1, "max_tokens": 8, "temperature": 0}
    status1, body1 = api(base_url, "POST", "/v1/chat/completions", payload1)
    if status1 != 200:
        return "fail", f"first request failed: {status1}"
    assistant_reply = body1["choices"][0]["message"]["content"]

    # Second request — appends the assistant reply + a new user message
    msg2 = msg1 + [
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": "Name another one."},
    ]
    payload2 = {"messages": msg2, "max_tokens": 8, "temperature": 0}
    status2, body2 = api(base_url, "POST", "/v1/chat/completions", payload2)
    if status2 != 200:
        return "fail", f"second request failed: {status2}"
    cached2 = body2["usage"]["cached_tokens"]

    if cached2 <= 0:
        return "fail", f"expected cache hit (cached_tokens > 0), got {cached2}"


def test_cache_miss_different_conversation(base_url: str, **_):
    """A completely different conversation causes a cache miss."""
    # First request
    payload1 = {
        "messages": [{"role": "user", "content": "Tell me about dogs."}],
        "max_tokens": 8,
    }
    api(base_url, "POST", "/v1/chat/completions", payload1)

    # Second request — totally different prompt, no shared prefix
    payload2 = {
        "messages": [{"role": "user", "content": "Tell me about quantum physics."}],
        "max_tokens": 8,
    }
    status, body = api(base_url, "POST", "/v1/chat/completions", payload2)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    cached = body["usage"]["cached_tokens"]
    if cached != 0:
        return "fail", f"expected cache miss (cached_tokens=0), got {cached}"


def test_cache_hit_four_turns(base_url: str, **_):
    """Four-turn conversation: each successive turn gets a cache hit."""
    messages = [{"role": "user", "content": "Count from 1 to 3. One number per line."}]

    # Turn 1 — cold start, no cache
    payload = {"messages": messages, "max_tokens": 16, "temperature": 0}
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"turn 1 failed: {status}"
    reply1 = body["choices"][0]["message"]["content"]
    cached1 = body["usage"]["cached_tokens"]

    prev_cached = cached1
    for turn, user_msg in enumerate(
        ["Now count from 4 to 6.", "Now 7 to 9.", "And finally 10."],
        start=2,
    ):
        messages += [
            {"role": "assistant", "content": reply1},
            {"role": "user", "content": user_msg},
        ]
        payload = {"messages": messages, "max_tokens": 16, "temperature": 0}
        status, body = api(base_url, "POST", "/v1/chat/completions", payload)
        if status != 200:
            return "fail", f"turn {turn} failed: {status}"
        reply1 = body["choices"][0]["message"]["content"]
        cached = body["usage"]["cached_tokens"]
        if cached <= 0:
            return "fail", f"turn {turn}: expected cache hit, got cached_tokens={cached}"
        if cached <= prev_cached:
            return "fail", (
                f"turn {turn}: cached_tokens should grow as conversation extends "
                f"(prev={prev_cached}, now={cached})"
            )
        prev_cached = cached


def test_cache_reset_then_resume(base_url: str, **_):
    """Interruption at turn 2 resets cache; turns 3-4 of the new conversation
    continue and turn 4 gets a cache hit."""
    # Turn 1 — conversation A
    msgs_a = [{"role": "user", "content": "What is the largest ocean?"}]
    payload = {"messages": msgs_a, "max_tokens": 16, "temperature": 0}
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"turn 1 failed: {status}"
    reply_a1 = body["choices"][0]["message"]["content"]
    msgs_a += [{"role": "assistant", "content": reply_a1}]

    # Turn 2 — interruption: completely different conversation B
    msgs_b = [{"role": "user", "content": "Name a planet in our solar system. One word."}]
    payload = {"messages": msgs_b, "max_tokens": 8, "temperature": 0}
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"turn 2 (interruption) failed: {status}"
    cached2 = body["usage"]["cached_tokens"]
    if cached2 != 0:
        return "fail", f"turn 2: expected cache miss after interruption, got {cached2}"
    reply_b1 = body["choices"][0]["message"]["content"]
    msgs_b += [{"role": "assistant", "content": reply_b1}]

    # Turn 3 — continue conversation B (cache hit against turn 2)
    msgs_b += [{"role": "user", "content": "Name another one."}]
    payload = {"messages": msgs_b, "max_tokens": 8, "temperature": 0}
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"turn 3 failed: {status}"
    cached3 = body["usage"]["cached_tokens"]
    if cached3 <= 0:
        return "fail", f"turn 3: expected cache hit continuing conversation B, got {cached3}"
    reply_b2 = body["choices"][0]["message"]["content"]
    msgs_b += [{"role": "assistant", "content": reply_b2}]

    # Turn 4 — continue conversation B again (cache hit against turn 3)
    msgs_b += [{"role": "user", "content": "One more."}]
    payload = {"messages": msgs_b, "max_tokens": 8, "temperature": 0}
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"turn 4 failed: {status}"
    cached4 = body["usage"]["cached_tokens"]
    if cached4 <= 0:
        return "fail", f"turn 4: expected cache hit, got {cached4}"
    if cached4 <= cached3:
        return "fail", (
            f"turn 4: cached_tokens should exceed turn 3 "
            f"(turn3={cached3}, turn4={cached4})"
        )


def test_cache_interleaved_clients(base_url: str, **_):
    """Simulate two clients interleaving: A, B, A-continue (miss), A-continue
    (hit), A-new-topic (miss), A-new-continue (hit)."""
    # Step 1 — Client A starts a conversation
    msgs_a = [{"role": "user", "content": "What is the tallest mountain?"}]
    s, body = api(base_url, "POST", "/v1/chat/completions",
                  {"messages": msgs_a, "max_tokens": 16, "temperature": 0})
    if s != 200:
        return "fail", f"step 1 (A) failed: {s}"
    reply_a1 = body["choices"][0]["message"]["content"]
    msgs_a += [{"role": "assistant", "content": reply_a1}]

    # Step 2 — Client B interrupts with its own conversation
    msgs_b = [{"role": "user", "content": "What is the deepest trench?"}]
    s, body = api(base_url, "POST", "/v1/chat/completions",
                  {"messages": msgs_b, "max_tokens": 16, "temperature": 0})
    if s != 200:
        return "fail", f"step 2 (B) failed: {s}"

    # Step 3 — Client A continues (miss: cache holds B's state)
    msgs_a += [{"role": "user", "content": "How tall is it?"}]
    s, body = api(base_url, "POST", "/v1/chat/completions",
                  {"messages": msgs_a, "max_tokens": 16, "temperature": 0})
    if s != 200:
        return "fail", f"step 3 (A continue) failed: {s}"
    cached3 = body["usage"]["cached_tokens"]
    if cached3 != 0:
        return "fail", f"step 3: expected miss after B interrupted, got {cached3}"
    reply_a2 = body["choices"][0]["message"]["content"]
    msgs_a += [{"role": "assistant", "content": reply_a2}]

    # Step 4 — Client A continues again (hit: cache now holds A's state)
    msgs_a += [{"role": "user", "content": "In which country?"}]
    s, body = api(base_url, "POST", "/v1/chat/completions",
                  {"messages": msgs_a, "max_tokens": 16, "temperature": 0})
    if s != 200:
        return "fail", f"step 4 (A continue) failed: {s}"
    cached4 = body["usage"]["cached_tokens"]
    if cached4 <= 0:
        return "fail", f"step 4: expected hit, got {cached4}"

    # Step 5 — Client A starts a brand-new conversation (miss)
    msgs_a2 = [{"role": "user", "content": "Name a famous painting."}]
    s, body = api(base_url, "POST", "/v1/chat/completions",
                  {"messages": msgs_a2, "max_tokens": 16, "temperature": 0})
    if s != 200:
        return "fail", f"step 5 (A new topic) failed: {s}"
    cached5 = body["usage"]["cached_tokens"]
    if cached5 != 0:
        return "fail", f"step 5: expected miss on new topic, got {cached5}"
    reply_a2_1 = body["choices"][0]["message"]["content"]
    msgs_a2 += [{"role": "assistant", "content": reply_a2_1}]

    # Step 6 — Client A continues the new conversation (hit)
    msgs_a2 += [{"role": "user", "content": "Who painted it?"}]
    s, body = api(base_url, "POST", "/v1/chat/completions",
                  {"messages": msgs_a2, "max_tokens": 16, "temperature": 0})
    if s != 200:
        return "fail", f"step 6 (A new continue) failed: {s}"
    cached6 = body["usage"]["cached_tokens"]
    if cached6 <= 0:
        return "fail", f"step 6: expected hit, got {cached6}"


def test_completions_cached_tokens(base_url: str, **_):
    """POST /v1/completions also reports cached_tokens."""
    payload = {
        "prompt": "Once upon a time in a land far away",
        "max_tokens": 8,
    }
    status, body = api(base_url, "POST", "/v1/completions", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    usage = body.get("usage", {})
    if "cached_tokens" not in usage:
        return "fail", "usage missing 'cached_tokens' field"


def test_cache_cleared_after_hot_swap(base_url: str, model_dir: str | None = None, **_):
    """After a model hot-swap, the cache is cleared (cached_tokens=0)."""
    if model_dir is None:
        return "skip", "no --model-dir provided"

    # Prime the cache
    msg = [{"role": "user", "content": "Hello there."}]
    api(base_url, "POST", "/v1/chat/completions",
        {"messages": msg, "max_tokens": 8})

    # Hot-swap
    swap_status, swap_body = api(base_url, "POST", "/v1/models/load",
                                  {"model_dir": model_dir}, timeout=600)
    if swap_status != 200:
        return "fail", f"hot-swap failed: {swap_status}: {swap_body}"

    # First request after swap — should be a cache miss
    status, body = api(base_url, "POST", "/v1/chat/completions",
                       {"messages": msg, "max_tokens": 8})
    if status != 200:
        return "fail", f"post-swap inference failed: {status}"
    cached = body["usage"]["cached_tokens"]
    if cached != 0:
        return "fail", f"expected cached_tokens=0 after hot-swap, got {cached}"


def test_load_model(base_url: str, model_dir: str | None = None, **_):
    """POST /v1/models/load re-loads the current model successfully."""
    if model_dir is None:
        return "skip", "no --model-dir provided"

    payload = {"model_dir": model_dir}
    status, body = api(base_url, "POST", "/v1/models/load", payload, timeout=600)
    if status != 200:
        return "fail", f"expected 200, got {status}: {body}"
    if body["status"] != "success":
        return "fail", f"expected success, got {body}"
    if body["recompiled"] is not False:
        return "fail", "expected recompiled=false (runtime config, no recompilation)"

    chat_payload = {
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
    }
    status2, body2 = api(base_url, "POST", "/v1/chat/completions", chat_payload)
    if status2 != 200:
        return "fail", f"post-swap inference failed: {status2}"
    if len(body2["choices"][0]["message"]["content"]) == 0:
        return "fail", "empty post-swap response"


def test_load_invalid_model(base_url: str, **_):
    """POST /v1/models/load with a nonexistent path returns 500."""
    payload = {"model_dir": "/tmp/nonexistent_model_dir_12345"}
    status, body = api(base_url, "POST", "/v1/models/load", payload)
    if status != 500:
        return "fail", f"expected 500 for invalid model, got {status}: {body}"


# ---------------------------------------------------------------------------
# LoRA adapter tests (require --adapter-dir with qmodel.lora)
# ---------------------------------------------------------------------------

def test_load_model_with_adapter(base_url: str, model_dir: str | None = None,
                                 adapter_dir: str | None = None, **_):
    """POST /v1/models/load with adapter_dir loads LoRA from a separate directory."""
    if model_dir is None:
        return "skip", "no --model-dir provided"
    if adapter_dir is None:
        return "skip", "no --adapter-dir provided"

    import os
    lora_path = os.path.join(adapter_dir, "qmodel.lora")
    if not os.path.exists(lora_path):
        return "skip", f"qmodel.lora not found in {adapter_dir}"

    payload = {"model_dir": model_dir, "adapter_dir": adapter_dir}
    status, body = api(base_url, "POST", "/v1/models/load", payload, timeout=600)
    if status != 200:
        return "fail", f"expected 200, got {status}: {body}"
    if body["status"] != "success":
        return "fail", f"expected success, got {body}"

    chat_payload = {
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
    }
    status2, body2 = api(base_url, "POST", "/v1/chat/completions", chat_payload)
    if status2 != 200:
        return "fail", f"inference with LoRA failed: {status2}"
    if len(body2["choices"][0]["message"]["content"]) == 0:
        return "fail", "empty response with LoRA"


def test_load_model_invalid_adapter(base_url: str, model_dir: str | None = None, **_):
    """POST /v1/models/load with a nonexistent adapter_dir returns 500."""
    if model_dir is None:
        return "skip", "no --model-dir provided"

    payload = {"model_dir": model_dir, "adapter_dir": "/tmp/nonexistent_adapter_dir_12345"}
    status, body = api(base_url, "POST", "/v1/models/load", payload, timeout=600)
    if status != 500:
        return "fail", f"expected 500 for invalid adapter_dir, got {status}: {body}"


def test_swap_adapter_to_no_adapter(base_url: str, model_dir: str | None = None,
                                    adapter_dir: str | None = None, **_):
    """Swap from adapter-enabled to no adapter on the same model."""
    if model_dir is None:
        return "skip", "no --model-dir provided"
    if adapter_dir is None:
        return "skip", "no --adapter-dir provided"

    import os
    lora_path = os.path.join(adapter_dir, "qmodel.lora")
    if not os.path.exists(lora_path):
        return "skip", f"qmodel.lora not found in {adapter_dir}"

    # Load with adapter
    payload1 = {"model_dir": model_dir, "adapter_dir": adapter_dir}
    status1, body1 = api(base_url, "POST", "/v1/models/load", payload1, timeout=600)
    if status1 != 200:
        return "fail", f"adapter load failed: {status1}: {body1}"

    # Swap to no adapter
    payload2 = {"model_dir": model_dir}
    status2, body2 = api(base_url, "POST", "/v1/models/load", payload2, timeout=600)
    if status2 != 200:
        return "fail", f"no-adapter swap failed: {status2}: {body2}"

    chat_payload = {
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
    }
    status3, body3 = api(base_url, "POST", "/v1/chat/completions", chat_payload)
    if status3 != 200:
        return "fail", f"post-swap inference failed: {status3}"
    if len(body3["choices"][0]["message"]["content"]) == 0:
        return "fail", "empty post-swap response"


def test_concurrent_swap_conflict(base_url: str, model_dir: str | None = None, **_):
    """Two rapid concurrent swaps: both should succeed (swap is near-instant, no recompilation)."""
    if model_dir is None:
        return "skip", "no --model-dir provided"

    results: list[dict] = [{}, {}]

    def _swap(idx):
        try:
            s, b = api(base_url, "POST", "/v1/models/load",
                       {"model_dir": model_dir}, timeout=600)
            results[idx]["status"] = s
            results[idx]["body"] = b
        except Exception as exc:
            results[idx]["error"] = str(exc)

    t1 = threading.Thread(target=_swap, args=(0,))
    t2 = threading.Thread(target=_swap, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=600)
    t2.join(timeout=600)

    for i, r in enumerate(results):
        if "error" in r:
            return "fail", f"swap {i} raised an exception: {r['error']}"
        # Both 200 (serialized) and 409 (conflict) are acceptable
        if r.get("status") not in (200, 409):
            return "fail", f"swap {i} unexpected status: {r}"

    # At least one must have succeeded
    if not any(r.get("status") == 200 for r in results):
        return "fail", f"no swap succeeded: {results}"


def test_inference_during_swap(base_url: str, model_dir: str | None = None, **_):
    """Swap is near-instant (no recompilation); verify it completes and inference works after."""
    if model_dir is None:
        return "skip", "no --model-dir provided"

    # Trigger a swap (near-instant since no compilation is needed)
    status, body = api(base_url, "POST", "/v1/models/load",
                       {"model_dir": model_dir}, timeout=600)
    if status != 200:
        return "fail", f"swap failed: {status}: {body}"

    # Model should be immediately available after swap returns
    m_status, m_body = api(base_url, "GET", "/v1/models", timeout=10)
    if m_status != 200:
        return "fail", f"expected 200 for /v1/models, got {m_status}"
    if len(m_body.get("data", [])) != 1:
        return "fail", f"expected 1 model after swap, got {m_body['data']}"

    # Inference should work immediately
    chat_payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
    }
    status2, _ = api(base_url, "POST", "/v1/chat/completions", chat_payload)
    if status2 != 200:
        return "fail", f"post-swap inference failed: {status2}"


# ---------------------------------------------------------------------------
# Voice pipeline tests (auto-skipped when server lacks --voice)
# ---------------------------------------------------------------------------

def test_voices_list(base_url: str, **_):
    """GET /v1/voices returns predefined voices."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    status, body = api(base_url, "GET", "/v1/voices")
    if status != 200:
        return "fail", f"expected 200, got {status}"
    voices = body["voices"]
    if len(voices) < 8:
        return "fail", f"expected >=8 predefined voices, got {len(voices)}"

    predefined = [v for v in voices if v["type"] == "predefined"]
    if len(predefined) != 8:
        return "fail", f"expected 8 predefined, got {len(predefined)}"

    names = {v["voice_id"] for v in predefined}
    for expected in ("alba", "marius", "javert", "jean",
                     "fantine", "cosette", "eponine", "azelma"):
        if expected not in names:
            return "fail", f"missing predefined voice: {expected}"


def test_tts_wav(base_url: str, **_):
    """POST /v1/audio/speech produces a valid WAV file."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    payload = {"input": "Hello world, this is a test."}
    status, data, _ = api_binary(base_url, "POST", "/v1/audio/speech", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    if not _is_valid_wav(data):
        return "fail", "response is not a valid WAV file"
    if _wav_sample_rate(data) != 24000:
        return "fail", f"expected 24kHz, got {_wav_sample_rate(data)}"
    if len(data) <= 200:
        return "fail", f"WAV too small ({len(data)} bytes), likely empty"


def test_tts_pcm(base_url: str, **_):
    """POST /v1/audio/speech with response_format=pcm returns raw PCM."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    payload = {"input": "Testing raw PCM output.", "response_format": "pcm"}
    status, data, _ = api_binary(base_url, "POST", "/v1/audio/speech", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    if data[:4] == b"RIFF":
        return "fail", "PCM response should not have WAV header"
    if len(data) <= 100:
        return "fail", f"PCM too small ({len(data)} bytes)"


def test_tts_voice_selection(base_url: str, **_):
    """POST /v1/audio/speech with an explicit voice produces audio."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    payload = {"input": "Voice selection test.", "voice": "jean"}
    status, data, _ = api_binary(base_url, "POST", "/v1/audio/speech", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    if not _is_valid_wav(data):
        return "fail", "response is not a valid WAV file"
    if len(data) <= 200:
        return "fail", f"WAV too small ({len(data)} bytes)"


def test_tts_empty_input(base_url: str, **_):
    """POST /v1/audio/speech with empty input returns 400."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    payload = {"input": "   "}
    status, _, _ = api_binary(base_url, "POST", "/v1/audio/speech", payload)
    if status != 400:
        return "fail", f"expected 400 for empty input, got {status}"


def test_stt_roundtrip(base_url: str, **_):
    """Generate WAV via TTS then transcribe it via STT."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    tts_payload = {"input": "The quick brown fox jumps over the lazy dog."}
    status, wav_data, _ = api_binary(base_url, "POST", "/v1/audio/speech",
                                     tts_payload)
    if status != 200:
        return "fail", f"TTS failed: {status}"
    if not _is_valid_wav(wav_data):
        return "fail", "TTS did not produce valid WAV"

    status, body, _ = multipart(
        base_url, "/v1/audio/transcriptions",
        fields={"model": "whisper-1"},
        files={"file": ("test.wav", wav_data, "audio/wav")},
    )
    if status != 200:
        return "fail", f"STT failed: {status}"
    result = json.loads(body)
    text = result.get("text", "")
    if len(text) == 0:
        return "fail", "STT returned empty transcription"


def test_stt_text_format(base_url: str, **_):
    """POST /v1/audio/transcriptions with response_format=text returns plain text."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    status, wav_data, _ = api_binary(base_url, "POST", "/v1/audio/speech",
                                     {"input": "Hello."})
    if status != 200:
        return "fail", f"TTS failed: {status}"

    status, body, _ = multipart(
        base_url, "/v1/audio/transcriptions",
        fields={"model": "whisper-1", "response_format": "text"},
        files={"file": ("test.wav", wav_data, "audio/wav")},
    )
    if status != 200:
        return "fail", f"STT failed: {status}"
    text = body.decode()
    if text.startswith("{"):
        return "fail", "expected plain text, got JSON"
    if len(text) == 0:
        return "fail", "STT returned empty text"


def test_custom_voice_lifecycle(base_url: str, **_):
    """Upload a custom voice, verify it appears, use it for TTS, then delete it."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    voice_id = f"test_voice_{uuid.uuid4().hex[:8]}"

    status, wav_data, _ = api_binary(base_url, "POST", "/v1/audio/speech",
                                     {"input": "This is my custom voice sample "
                                               "for cloning purposes."})
    if status != 200:
        return "fail", f"TTS failed: {status}"

    status, body, _ = multipart(
        base_url, "/v1/voices",
        fields={"voice_id": voice_id},
        files={"file": (f"{voice_id}.wav", wav_data, "audio/wav")},
    )
    if status != 200:
        return "fail", f"voice upload failed: {status} — {body.decode()}"
    result = json.loads(body)
    if result["voice_id"] != voice_id:
        return "fail", f"expected voice_id={voice_id}, got {result['voice_id']}"
    if result["status"] != "created":
        return "fail", f"expected status=created, got {result['status']}"

    status, body = api(base_url, "GET", "/v1/voices")
    if status != 200:
        return "fail", f"GET /v1/voices failed: {status}"
    voices = body["voices"]
    custom = [v for v in voices if v["voice_id"] == voice_id]
    if len(custom) != 1:
        return "fail", "custom voice not found in list"
    if custom[0]["type"] != "custom":
        return "fail", f"expected type=custom, got {custom[0]['type']}"

    payload = {"input": "Testing custom voice.", "voice": voice_id}
    status, audio, _ = api_binary(base_url, "POST", "/v1/audio/speech", payload)
    if status != 200:
        return "fail", f"TTS with custom voice failed: {status}"
    if not _is_valid_wav(audio):
        return "fail", "custom voice TTS did not produce valid WAV"

    url = f"{base_url}/v1/voices/{voice_id}"
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            del_status = resp.status
            del_body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        del_status = exc.code
        del_body = {}
    if del_status != 200:
        return "fail", f"delete failed: {del_status}"
    if del_body.get("status") != "deleted":
        return "fail", f"expected status=deleted, got {del_body.get('status')}"

    status, body = api(base_url, "GET", "/v1/voices")
    if status != 200:
        return "fail", f"GET /v1/voices after delete failed: {status}"
    remaining = [v for v in body["voices"] if v["voice_id"] == voice_id]
    if len(remaining) != 0:
        return "fail", "custom voice still in list after deletion"


def test_custom_voice_predefined_conflict(base_url: str, **_):
    """Uploading a voice with a predefined name returns 400."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    dummy_wav = b"RIFF" + b"\x00" * 40

    status, _, _ = multipart(
        base_url, "/v1/voices",
        fields={"voice_id": "alba"},
        files={"file": ("alba.wav", dummy_wav, "audio/wav")},
    )
    if status != 400:
        return "fail", f"expected 400 for predefined voice name, got {status}"


def test_delete_nonexistent_voice(base_url: str, **_):
    """Deleting a voice that doesn't exist returns 404."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    url = f"{base_url}/v1/voices/nonexistent_voice_xyz"
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    if status != 404:
        return "fail", f"expected 404, got {status}"


def test_delete_predefined_voice(base_url: str, **_):
    """Deleting a predefined voice returns 400."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    url = f"{base_url}/v1/voices/alba"
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    if status != 400:
        return "fail", f"expected 400, got {status}"


def test_chat_still_works_with_voice(base_url: str, **_):
    """Existing /v1/chat/completions still works when voice pipeline is active."""
    if not _voice_enabled(base_url):
        return "skip", "voice pipeline not enabled"

    payload = {
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
    }
    status, body = api(base_url, "POST", "/v1/chat/completions", payload)
    if status != 200:
        return "fail", f"expected 200, got {status}"
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str) or len(content) == 0:
        return "fail", "empty response"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_models_endpoint,
    test_chat_non_streaming,
    test_chat_streaming,
    test_completions_non_streaming,
    test_completions_streaming,
    test_greedy_decoding,
    test_max_tokens,
    # KV cache persistence tests
    test_cached_tokens_field,
    test_cache_hit_multi_turn,
    test_cache_miss_different_conversation,
    test_cache_hit_four_turns,
    test_cache_reset_then_resume,
    test_cache_interleaved_clients,
    test_completions_cached_tokens,
    test_cache_cleared_after_hot_swap,
    # Hot-swap and LoRA tests
    test_load_model,
    test_concurrent_swap_conflict,
    test_inference_during_swap,
    test_load_invalid_model,
    # LoRA adapter tests (require --adapter-dir with qmodel.lora; skip gracefully if missing)
    test_load_model_with_adapter,
    test_load_model_invalid_adapter,
    test_swap_adapter_to_no_adapter,
]

VOICE_TESTS = [
    test_voices_list,
    test_tts_wav,
    test_tts_pcm,
    test_tts_voice_selection,
    test_tts_empty_input,
    test_stt_roundtrip,
    test_stt_text_format,
    test_custom_voice_lifecycle,
    test_custom_voice_predefined_conflict,
    test_delete_nonexistent_voice,
    test_delete_predefined_voice,
    test_chat_still_works_with_voice,
]


def main():
    parser = argparse.ArgumentParser(description="Trillim API server tests")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="Server base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Model directory for the load-model test",
    )
    parser.add_argument(
        "--adapter-dir", default=None,
        help="LoRA adapter directory (separate from model dir) for adapter tests",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    # Check server is reachable
    try:
        api(base, "GET", "/v1/models", timeout=5)
    except Exception:
        print(f"Could not connect to {base}")
        print("Start the server first:  make serve MODEL_DIR=<path>")
        sys.exit(1)

    all_tests = list(ALL_TESTS) + list(VOICE_TESTS)

    passed = 0
    failed = 0
    skipped = 0

    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    for test_fn in all_tests:
        name = test_fn.__name__
        try:
            result = test_fn(base_url=base, model_dir=args.model_dir, adapter_dir=args.adapter_dir)
            if isinstance(result, tuple):
                status, msg = result
                if status == "fail":
                    failed += 1
                    print(f"  {RED}FAIL{RESET}  {name}: {msg}")
                elif status == "skip":
                    skipped += 1
                    print(f"  {YELLOW}SKIP{RESET}  {name}: {msg}")
            else:
                passed += 1
                print(f"  {GREEN}PASS{RESET}  {name}")
        except Exception as exc:
            failed += 1
            print(f"  {RED}FAIL{RESET}  {name}: {exc}")

    print()
    print(f"Results: {GREEN}{passed} passed{RESET}, "
          f"{RED}{failed} failed{RESET}, "
          f"{YELLOW}{skipped} skipped{RESET}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
