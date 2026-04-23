"""Test-only child-process bootstrap for E2E server runs."""

from __future__ import annotations

import asyncio
import os
from functools import wraps
from pathlib import Path


def _start_subprocess_coverage() -> None:
    config_path = os.environ.get("COVERAGE_PROCESS_START")
    if not config_path:
        return
    try:
        import coverage
    except Exception:
        return
    coverage.process_startup()


def _install_e2e_patches() -> None:
    if os.environ.get("TRILLIM_E2E_CHILD") != "1":
        return

    import trillim.cli as cli
    import trillim.components.stt.public as stt_public
    import trillim.components.tts.public as tts_public
    from tests.components.llm.support import FakeTokenizer, make_runtime_model
    from tests.components.tts.support import (
        FakeTokenizer as FakeTTSTokenizer,
        fake_voice_state_builder,
    )
    from trillim.components.llm.public import LLM
    from trillim.components.tts.public import TTS

    port = os.environ.get("TRILLIM_E2E_PORT")
    if port is not None:
        cli.DEFAULT_PORT = int(port)
    cli.DEFAULT_HOST = "127.0.0.1"

    if not getattr(LLM.__init__, "_trillim_e2e_patched", False):
        original_llm_init = LLM.__init__

        class _E2EFakeEngine:
            def __init__(
                self,
                model,
                tokenizer,
                defaults,
                *,
                init_config=None,
                progress_timeout: float,
            ) -> None:
                self.model = model
                self.tokenizer = tokenizer
                self.defaults = defaults
                self.init_config = init_config
                self.progress_timeout = progress_timeout
                self._cached_token_ids: list[int] = []
                self._last_cache_hit = 0
                self._last_prompt_tokens = 0
                self._last_completion_tokens = 0

            @property
            def cached_token_ids(self) -> list[int]:
                return list(self._cached_token_ids)

            @property
            def cached_token_count(self) -> int:
                return len(self._cached_token_ids)

            @property
            def last_cache_hit(self) -> int:
                return self._last_cache_hit

            @property
            def last_prompt_tokens(self) -> int:
                return self._last_prompt_tokens

            @property
            def last_completion_tokens(self) -> int:
                return self._last_completion_tokens

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                self._cached_token_ids = []
                self._last_cache_hit = 0
                self._last_prompt_tokens = 0
                self._last_completion_tokens = 0

            async def generate(self, token_ids, **_sampling):
                request_tokens = list(token_ids)
                shared = 0
                limit = min(len(request_tokens), len(self._cached_token_ids))
                while (
                    shared < limit
                    and request_tokens[shared] == self._cached_token_ids[shared]
                ):
                    shared += 1
                self._last_cache_hit = shared if shared == len(self._cached_token_ids) else 0

                prompt_text = self.tokenizer.decode(
                    request_tokens,
                    skip_special_tokens=True,
                )
                if "slow" in prompt_text.lower():
                    await asyncio.sleep(
                        float(
                            os.environ.get(
                                "TRILLIM_E2E_SLOW_DELAY_SECONDS",
                                "1.0",
                            )
                        )
                    )
                response = (
                    os.environ.get("TRILLIM_E2E_STREAM_RESPONSE", "streamed reply")
                    if "stream" in prompt_text.lower()
                    else os.environ.get("TRILLIM_E2E_RESPONSE_TEXT", "hello")
                )
                generated = [ord(character) for character in response]
                for token_id in generated:
                    yield token_id
                self._cached_token_ids = request_tokens + generated
                self._last_prompt_tokens = len(request_tokens)
                self._last_completion_tokens = len(generated)

        def _fake_model_validator(model_dir):
            path = Path(model_dir)
            return make_runtime_model(path, name=path.name)

        @wraps(original_llm_init)
        def patched_llm_init(self, model_dir, **kwargs):
            kwargs.setdefault("_model_validator", _fake_model_validator)
            kwargs.setdefault("_tokenizer_loader", lambda *_args, **_kwargs: FakeTokenizer())
            kwargs.setdefault("_engine_factory", _E2EFakeEngine)
            return original_llm_init(self, model_dir, **kwargs)

        patched_llm_init._trillim_e2e_patched = True  # type: ignore[attr-defined]
        LLM.__init__ = patched_llm_init

    async def fake_stt_engine_start(self):
        return None

    async def fake_stt_engine_stop(self):
        return None

    async def fake_stt_engine_transcribe(self, pcm, *, conditioning_text="", language=None):
        del conditioning_text
        text = bytes(pcm).decode("latin-1")
        return text if language is None else f"{language}:{text}"

    stt_public.STTEngine.start = fake_stt_engine_start
    stt_public.STTEngine.stop = fake_stt_engine_stop
    stt_public.STTEngine.transcribe = fake_stt_engine_transcribe

    if not getattr(TTS.__init__, "_trillim_e2e_patched", False):
        original_tts_init = TTS.__init__

        class _FakeSessionWorker:
            def __init__(self, *, voice_kind: str, voice_reference: str) -> None:
                self._voice_kind = voice_kind
                self._voice_reference = voice_reference

            async def synthesize(self, text: str) -> bytes:
                return (
                    f"{self._voice_kind}:{self._voice_reference}:{text}"
                ).encode("utf-8")

            async def close(self) -> None:
                return None

        @wraps(original_tts_init)
        def patched_tts_init(self, *args, **kwargs):
            kwargs.setdefault("_tokenizer_loader", lambda: FakeTTSTokenizer())
            kwargs.setdefault(
                "_session_worker_factory",
                lambda *, voice_kind, voice_reference: _FakeSessionWorker(
                    voice_kind=voice_kind,
                    voice_reference=voice_reference,
                ),
            )
            kwargs.setdefault("_voice_state_builder", fake_voice_state_builder)
            return original_tts_init(self, *args, **kwargs)

        patched_tts_init._trillim_e2e_patched = True  # type: ignore[attr-defined]
        TTS.__init__ = patched_tts_init

    original_tts_import = tts_public.importlib.import_module

    def patched_tts_import(module_name: str):
        if module_name in {"numpy", "soundfile", "pocket_tts"}:
            return object()
        return original_tts_import(module_name)

    tts_public.importlib.import_module = patched_tts_import
    tts_public._load_built_in_voice_names = lambda: ("alba", "marius")


_start_subprocess_coverage()
_install_e2e_patches()
