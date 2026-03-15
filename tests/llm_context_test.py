# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for ChatSession prompt metrics and validation."""

from types import SimpleNamespace
import unittest

import trillim
from trillim import ChatSession, ContextOverflowError
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


class _PatternMergingTokenizer(_TrackingTokenizer):
    def __init__(self, patterns: list[str]):
        super().__init__()
        self._patterns = sorted(enumerate(patterns), key=lambda item: len(item[1]), reverse=True)

    def encode(self, text: str, add_special_tokens: bool = True):
        self.encode_calls.append((text, add_special_tokens))
        token_ids: list[int] = []
        index = 0
        while index < len(text):
            for pattern_index, pattern in self._patterns:
                if text.startswith(pattern, index):
                    token_ids.append(1000 + pattern_index)
                    index += len(pattern)
                    break
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return token_ids


class _EotTemplateTokenizer(_TrackingTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}<|eot_id|>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class _ScriptedEngine:
    def __init__(self, responses: list[str], tokenizer, *, max_context_tokens: int = 128):
        self._responses = list(responses)
        self.tokenizer = tokenizer
        self.arch_config = SimpleNamespace(max_position_embeddings=max_context_tokens)
        self._cached_token_ids: list[int] = []
        self._cached_prompt_str = ""
        self._last_cache_hit = 0
        self.finalized_prompt_snapshots = []

    @property
    def cached_token_ids(self) -> list[int]:
        return list(self._cached_token_ids)

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

    async def test_session_returns_public_chat_session_type(self):
        llm, _ = _make_llm()
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertIs(type(session), ChatSession)

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

    async def test_session_reuses_generated_tokens_without_full_prompt_reencode(self):
        llm, tokenizer = _make_llm(responses=["ok", "again"], max_context_tokens=2048)
        user_text = "h" * 700
        session = llm.session([{"role": "user", "content": user_text}])
        assistant_prompt = f"<user>{user_text}</user><assistant>"

        self.assertEqual(session.prompt_tokens, len(assistant_prompt))
        self.assertEqual(await session.chat(), "ok")
        self.assertNotIn(
            assistant_prompt,
            [text for text, _ in tokenizer.encode_calls[1:]],
        )
        self.assertEqual(
            session.messages,
            (
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "ok"},
            ),
        )

        encode_count_after_reply = len(tokenizer.encode_calls)
        session.add_user("next")
        next_prompt_tokens = session.prompt_tokens
        next_prompt = f"{assistant_prompt}ok</assistant><user>next</user><assistant>"
        new_call_texts = [text for text, _ in tokenizer.encode_calls[encode_count_after_reply:]]

        self.assertGreater(next_prompt_tokens, 0)
        self.assertIn("<user>next</user>", new_call_texts)
        self.assertIn("<assistant>", new_call_texts)
        self.assertNotIn(
            next_prompt,
            new_call_texts,
        )

    async def test_session_checks_multiple_overlap_windows_before_staying_on_fast_path(self):
        user_text = "h" * 700
        llm, tokenizer = _make_llm(max_context_tokens=2048)
        session = llm.session([{"role": "user", "content": user_text}])
        encode_count_after_init = len(tokenizer.encode_calls)

        self.assertEqual(session.prompt_tokens, len(f"<user>{user_text}</user><assistant>"))
        self.assertEqual(
            [text for text, _ in tokenizer.encode_calls[encode_count_after_init:]],
            [
                "<assistant>",
                session._base_prompt_str[-8:],
                session._base_prompt_str[-8:] + "<assistant>",
                session._base_prompt_str[-16:],
                session._base_prompt_str[-16:] + "<assistant>",
                session._base_prompt_str[-32:],
                session._base_prompt_str[-32:] + "<assistant>",
            ],
        )

    async def test_session_short_circuits_empty_suffix_overlap_checks(self):
        llm, _ = _make_llm()
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(session._encode_suffix(""), [])
        self.assertTrue(session._suffix_passes_overlap_validation("", "tail", [1, 2, 3]))
        self.assertTrue(session._suffix_passes_overlap_validation("base", "", []))

    async def test_session_prompt_edits_do_not_mutate_backend_cache_before_generation(self):
        llm, _ = _make_llm(responses=["ok", "again"])
        llm.engine._cached_prompt_str = "previous-cache"
        session = llm.session([{"role": "user", "content": "hello"}])

        self.assertEqual(llm.engine.cached_prompt_str, "previous-cache")

        await session.chat()
        self.assertEqual(llm.engine.cached_prompt_str, "<user>hello</user><assistant>ok</assistant>")

        session.add_user("again")
        self.assertEqual(llm.engine.cached_prompt_str, "<user>hello</user><assistant>ok</assistant>")

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

    async def test_session_full_reencodes_when_checked_overlap_window_detects_boundary_merge(self):
        merge_pattern = ("x" * 20) + "</user><assistant>"
        user_text = ("p" * 120) + ("x" * 20)
        tokenizer = _PatternMergingTokenizer([merge_pattern])
        llm, _ = _make_llm(tokenizer=tokenizer, max_context_tokens=2048)
        session = llm.session([{"role": "user", "content": user_text}])
        encode_count_after_init = len(tokenizer.encode_calls)
        assistant_prompt = f"<user>{user_text}</user><assistant>"
        prompt_tokens = session.prompt_tokens

        self.assertIn(
            assistant_prompt,
            [text for text, _ in tokenizer.encode_calls[encode_count_after_init:]],
        )
        self.assertLess(prompt_tokens, len(assistant_prompt))

    async def test_session_finalization_full_reencodes_when_tail_boundary_is_not_safe(self):
        assistant_text = ("p" * 120) + ("x" * 20)
        tokenizer = _PatternMergingTokenizer([("x" * 20) + "</assistant>"])
        llm, _ = _make_llm(tokenizer=tokenizer, responses=[assistant_text])
        session = llm.session([{"role": "user", "content": "hello"}])
        final_prompt = f"<user>hello</user><assistant>{assistant_text}</assistant>"

        self.assertEqual(await session.chat(), assistant_text)
        self.assertIn(final_prompt, [text for text, _ in tokenizer.encode_calls])
        self.assertEqual(
            session._base_token_ids,
            tokenizer.encode(final_prompt, add_special_tokens=False),
        )

    async def test_session_finalization_snapshots_full_rendered_prompt_for_eot_templates(self):
        tokenizer = _EotTemplateTokenizer()
        llm, _ = _make_llm(tokenizer=tokenizer, responses=["ok"])
        session = llm.session([{"role": "user", "content": "hello"}])
        final_prompt = "<user>hello<|eot_id|><assistant>ok<|eot_id|>"

        self.assertEqual(await session.chat(), "ok")
        self.assertEqual(session._base_prompt_str, final_prompt)
        self.assertEqual(session._base_token_ids, tokenizer.encode(final_prompt, add_special_tokens=False))
        self.assertEqual(len(llm.engine.finalized_prompt_snapshots), 1)
        self.assertEqual(llm.engine.finalized_prompt_snapshots[0].prompt_str, final_prompt)
        self.assertEqual(
            llm.engine.finalized_prompt_snapshots[0].token_ids,
            tuple(tokenizer.encode(final_prompt, add_special_tokens=False)),
        )

    async def test_session_finalization_preserves_truncated_cached_prefix_for_eot_templates(self):
        tokenizer = _EotTemplateTokenizer()
        llm, _ = _make_llm(tokenizer=tokenizer, responses=["ok"])
        session = llm.session([{"role": "user", "content": "hello"}])
        prepared_token_ids, prepared_prompt = session._prepare_reply()
        llm.engine._cached_token_ids = prepared_token_ids + tokenizer.encode("ok", add_special_tokens=False)

        self.assertEqual(await session.chat(), "ok")
        self.assertEqual(len(llm.engine.finalized_prompt_snapshots), 1)
        self.assertEqual(
            llm.engine.finalized_prompt_snapshots[0].prompt_str,
            f"{prepared_prompt}ok",
        )
        self.assertEqual(
            llm.engine.finalized_prompt_snapshots[0].token_ids,
            tuple(prepared_token_ids + tokenizer.encode("ok", add_special_tokens=False)),
        )


class ChatSessionLifecycleTests(unittest.TestCase):
    def test_session_api_requires_started_llm(self):
        llm = LLM("models/fake")

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            llm.session([{"role": "user", "content": "hello"}])

        with self.assertRaisesRegex(RuntimeError, "LLM not started"):
            _ = llm.max_context_tokens


if __name__ == "__main__":
    unittest.main()
