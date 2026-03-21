"""Test helpers for Phase 2 LLM tests."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from trillim.components.llm._config import (
    ActivationType,
    ArchitectureType,
    ModelRuntimeConfig,
    SamplingDefaults,
)
from trillim.components.llm._engine import EngineCrashedError, EngineProgressTimeoutError


class FakeTokenizer:
    """Minimal tokenizer stub for LLM tests."""

    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return text


class FakeEngine:
    """Scripted fake engine used by session, harness, router, and public tests."""

    def __init__(
        self,
        model: ModelRuntimeConfig,
        tokenizer,
        defaults: SamplingDefaults,
        *,
        responses: Sequence[str] | None = None,
        kv_positions: Sequence[int | None] | None = None,
        failure: Exception | None = None,
        start_error: Exception | None = None,
        lifecycle_log: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.defaults = defaults
        self.responses = list(responses or [])
        self.kv_positions = list(kv_positions or [])
        self.failure = failure
        self.start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0
        self.generate_calls: list[dict] = []
        self.lifecycle_log = lifecycle_log
        self.process = object()
        self._cached_token_ids: list[int] = []
        self._last_cache_hit = 0

    @property
    def cached_token_ids(self) -> list[int]:
        return list(self._cached_token_ids)

    @property
    def cached_token_count(self) -> int:
        return len(self._cached_token_ids)

    @property
    def last_cache_hit(self) -> int:
        return self._last_cache_hit

    async def start(self) -> None:
        self.start_calls += 1
        if self.lifecycle_log is not None:
            self.lifecycle_log.append(f"start:{self.model.name}")
        if self.start_error is not None:
            raise self.start_error

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.lifecycle_log is not None:
            self.lifecycle_log.append(f"stop:{self.model.name}")
        self.process = None

    async def generate(self, token_ids, **sampling):
        self.generate_calls.append({"token_ids": list(token_ids), **sampling})
        if self.failure is not None:
            raise self.failure
        request_tokens = list(token_ids)
        shared = 0
        limit = min(len(request_tokens), len(self._cached_token_ids))
        while shared < limit and request_tokens[shared] == self._cached_token_ids[shared]:
            shared += 1
        self._last_cache_hit = shared if shared == len(self._cached_token_ids) else 0
        response = self.responses.pop(0) if self.responses else ""
        generated = [ord(character) for character in response]
        for token_id in generated:
            yield token_id
        kv_position = (
            self.kv_positions.pop(0)
            if self.kv_positions
            else len(request_tokens) + len(generated)
        )
        if kv_position is None:
            kv_position = len(request_tokens) + len(generated)
        combined = request_tokens + generated
        if kv_position > len(combined):
            combined = combined + ([0] * (kv_position - len(combined)))
        self._cached_token_ids = combined[:kv_position]


class FakeEngineFactory:
    """Factory helper that records created fake engines."""

    def __init__(
        self,
        *,
        responses: Sequence[str] | None = None,
        kv_positions: Sequence[int | None] | None = None,
        failure: Exception | None = None,
        start_error: Exception | None = None,
        lifecycle_log: list[str] | None = None,
    ) -> None:
        self.responses = responses
        self.kv_positions = kv_positions
        self.failure = failure
        self.start_error = start_error
        self.lifecycle_log = lifecycle_log
        self.instances: list[FakeEngine] = []

    def __call__(self, model, tokenizer, defaults, **kwargs):
        engine = FakeEngine(
            model,
            tokenizer,
            defaults,
            responses=self.responses,
            kv_positions=self.kv_positions,
            failure=self.failure,
            start_error=self.start_error,
            lifecycle_log=self.lifecycle_log,
            **kwargs,
        )
        self.instances.append(engine)
        return engine


def make_runtime_model(path: Path, name: str | None = None) -> ModelRuntimeConfig:
    """Build a minimal validated runtime model config."""
    return ModelRuntimeConfig(
        name=name or path.name,
        path=path,
        arch_type=ArchitectureType.LLAMA,
        activation=ActivationType.SILU,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        vocab_size=1024,
        head_dim=64,
        max_position_embeddings=4096,
        norm_eps=1e-6,
        rope_theta=10000.0,
        eos_tokens=(0,),
        has_qkv_bias=False,
        tie_word_embeddings=False,
        has_attn_sub_norm=False,
        has_ffn_sub_norm=False,
    )


@contextmanager
def model_dir(
    *,
    architecture: str = "LlamaForCausalLM",
    hidden_act: str = "silu",
    tokenizer_payload: dict | None = None,
    text_config: dict | None = None,
    extra_config: dict | None = None,
):
    """Create a temporary on-disk model directory for validation tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config = {
            "architectures": [architecture],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4,
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "hidden_act": hidden_act,
            "eos_token_id": 2,
        }
        if text_config is not None:
            config = {"text_config": text_config, "architectures": [architecture]}
        if extra_config:
            config.update(extra_config)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        tokenizer_json = tokenizer_payload or {
            "added_tokens": [{"content": "</s>", "id": 2}]
        }
        (root / "tokenizer.json").write_text(
            json.dumps(tokenizer_json),
            encoding="utf-8",
        )
        (root / "qmodel.tensors").write_bytes(b"quantized-model")
        (root / "rope.cache").write_bytes(b"rope-cache")
        yield root


def progress_timeout() -> EngineProgressTimeoutError:
    """Return a reusable fake progress-timeout error."""
    return EngineProgressTimeoutError("Inference engine made no token_id progress for 5.0 seconds")


def crashed() -> EngineCrashedError:
    """Return a reusable fake engine-crash error."""
    return EngineCrashedError("Inference engine crashed: boom")
