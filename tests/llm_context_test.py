# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for ChatSession prompt metrics and validation."""

from types import SimpleNamespace
import unittest

import trillim
from trillim import ContextOverflowError
from trillim.harnesses._default import DefaultHarness
from trillim.server import LLM
from trillim.server._models import ServerState


class _TrackingTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self):
        self.encode_calls: list[tuple[str, bool]] = []

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def encode(self, text: str, add_special_tokens: bool = True):
        self.encode_calls.append((text, add_special_tokens))
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(chr(token_id) for token_id in token_ids)


class _RewritingTokenizer(_TrackingTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if not add_generation_prompt and len(messages) > 1:
            return f"rewritten::{rendered}"
        return rendered


class _ScriptedEngine:
    def __init__(self, responses: list[str], tokenizer, *, max_context_tokens: int = 128):
        self._responses = list(responses)
        self.tokenizer = tokenizer
        self.arch_config = SimpleNamespace(max_position_embeddings=max_context_tokens)
        self._cached_prompt_str = ""
        self._last_cache_hit = 0
        self.finalized_prompt_snapshots = []

    @property
    def cached_prompt_str(self) -> str:
        return self._cached_prompt_str

    @property
    def last_cache_hit(self) -> int:
        return self._last_cache_hit

    def finalize_prompt_cache(self, snapshot) -> None:
        self.finalized_prompt_snapshots.append(snapshot)
        self._cached_prompt_str = snapshot.prompt_str or ""

    def reset_prompt_cache(self) -> None:
        self._cached_prompt_str = ""
        self._last_cache_hit = 0

    async def generate(self, **_):
        response = self._responses.pop(0)
        for ch in response:
            yield ord(ch)


def _make_llm(
    tokenizer=None,
    *,
    responses: list[str] | None = None,
    max_context_tokens: int = 128,
) -> tuple[LLM, _TrackingTokenizer]:
    tokenizer = tokenizer or _TrackingTokenizer()
    llm = LLM("models/fake")
    llm.state = ServerState.RUNNING
    llm.engine = _ScriptedEngine(
        responses or [],
        tokenizer,
        max_context_tokens=max_context_tokens,
    )
    llm.harness = DefaultHarness(llm.engine)
    return llm, tokenizer


class ChatSessionMetricTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_overflow_error_is_exported(self):
        self.assertIs(trillim.ContextOverflowError, ContextOverflowError)

    async def test_session_exposes_prompt_metrics_without_duplicate_preparation(self):
        llm, tokenizer = _make_llm()
        session = llm.session([{"role": "system", "content": "rules"}])
        session.add_user("hello")

        prompt_tokens = session.prompt_tokens
        encode_calls_after_first_prepare = list(tokenizer.encode_calls)

        self.assertEqual(session.prompt_tokens, prompt_tokens)
        self.assertEqual(session.validate(), prompt_tokens)
        self.assertEqual(session.max_context_tokens, 128)
        self.assertEqual(session.remaining_context_tokens, 128 - prompt_tokens)
        self.assertEqual(tokenizer.encode_calls, encode_calls_after_first_prepare)
        self.assertEqual(session.messages[-1], {"role": "user", "content": "hello"})

    async def test_session_reuses_generated_tokens_and_only_encodes_new_suffixes(self):
        llm, tokenizer = _make_llm(responses=["ok", "again"])
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(session.prompt_tokens, len(tokenizer.encode_calls[0][0]) + len("<assistant>"))
        self.assertEqual(await session.chat(), "ok")
        self.assertEqual(tokenizer.encode_calls[-1], ("</assistant>", False))
        self.assertEqual(
            session.messages,
            (
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
            ),
        )

        encode_count_after_reply = len(tokenizer.encode_calls)
        session.add_user("next")
        next_prompt_tokens = session.prompt_tokens

        self.assertGreater(next_prompt_tokens, 0)
        self.assertEqual(
            tokenizer.encode_calls[encode_count_after_reply:],
            [
                ("<user>next</user>", False),
                ("<assistant>", False),
            ],
        )

    async def test_session_prompt_edits_do_not_mutate_backend_cache_before_generation(self):
        llm, _ = _make_llm(responses=["ok", "again"])
        llm.engine._cached_prompt_str = "previous-cache"
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(llm.engine.cached_prompt_str, "previous-cache")

        await session.chat()
        self.assertEqual(llm.engine.cached_prompt_str, "<user>hello</user><assistant>ok")

        session.add_user("again")
        self.assertEqual(llm.engine.cached_prompt_str, "<user>hello</user><assistant>ok")

    async def test_session_validate_raises_typed_overflow_error(self):
        llm, _ = _make_llm(max_context_tokens=8)
        session = llm.session([{"role": "user", "content": "too long"}])

        with self.assertRaises(ContextOverflowError) as ctx:
            session.validate()

        self.assertEqual(ctx.exception.max_context_tokens, 8)
        self.assertIn("exceeds context window", str(ctx.exception))

    async def test_session_requires_turn_ready_state(self):
        llm, _ = _make_llm(responses=["ok", "again"])
        empty = llm.session()

        with self.assertRaisesRegex(ValueError, "no messages"):
            _ = empty.prompt_tokens

        session = llm.session([{"role": "user", "content": "hello"}])
        self.assertEqual(await session.chat(), "ok")

        with self.assertRaisesRegex(ValueError, "assistant reply"):
            session.validate()

        session.add_user("again")
        self.assertEqual(await session.chat(), "again")

    async def test_session_rejects_stale_model_changes(self):
        llm, _ = _make_llm()
        session = llm.session([{"role": "user", "content": "hello"}])

        replacement = _ScriptedEngine(["new"], _TrackingTokenizer())
        llm.engine = replacement
        llm.harness = DefaultHarness(replacement)
        llm._session_generation += 1

        with self.assertRaisesRegex(RuntimeError, "stale"):
            _ = session.messages

    async def test_session_rejects_prompt_rewrites_when_appending_messages(self):
        llm, _ = _make_llm(tokenizer=_RewritingTokenizer())
        session = llm.session([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "append-only prompt rendering"):
            session.add_system("rules")


class ChatSessionLifecycleTests(unittest.TestCase):
    def test_session_api_requires_started_llm(self):
        llm = LLM("models/fake")

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            llm.session([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            _ = llm.max_context_tokens


if __name__ == "__main__":
    unittest.main()
