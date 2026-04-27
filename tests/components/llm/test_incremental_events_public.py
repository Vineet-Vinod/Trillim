from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim import _model_store
from trillim.components.llm._events import (
    ChatDoneEvent,
    ChatFinalTextEvent,
    ChatTokenEvent,
    ChatUsage,
)
from trillim.components.llm._incremental_decode import IncrementalDecoder
from trillim.components.llm._session import ChatSession
from trillim.components.llm.public import LLM, _make_init_config, load_tokenizer
from trillim.errors import ComponentLifecycleError, InvalidRequestError
from trillim.errors import ModelValidationError

from tests.support import write_llm_bundle


BONSAI_MODEL_ID = "Trillim/Bonsai-1.7B-TRNQ"
BONSAI_MODEL_DIR = _model_store.store_path_for_id(BONSAI_MODEL_ID)


@unittest.skipUnless(
    BONSAI_MODEL_DIR.is_dir(),
    f"{BONSAI_MODEL_ID} must be installed in the Trillim model store",
)
class IncrementalDecoderTests(unittest.TestCase):
    def test_decoder_streams_suffixes_from_real_bonsai_tokenizer_and_resets(self):
        tokenizer = load_tokenizer(BONSAI_MODEL_DIR, trust_remote_code=False)
        token_ids = tokenizer.encode("Hello world", add_special_tokens=False)
        self.assertGreaterEqual(len(token_ids), 2)
        decoder = IncrementalDecoder(tokenizer)

        chunks = [decoder.decode(token_id) for token_id in token_ids]
        self.assertEqual(
            "".join(chunks) + decoder.flush(),
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )

    def test_decoder_streams_real_unicode_and_emoji_sequences(self):
        tokenizer = load_tokenizer(BONSAI_MODEL_DIR, trust_remote_code=False)
        text = "Hello 👋🏽 café 你好 🚀"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoder = IncrementalDecoder(tokenizer)

        chunks = [decoder.decode(token_id) for token_id in token_ids]
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.assertEqual("".join(chunks) + decoder.flush(), decoded)
        self.assertIn("👋", decoded)
        self.assertIn("🚀", decoded)

    def test_decoder_compacts_long_real_unicode_streams(self):
        tokenizer = load_tokenizer(BONSAI_MODEL_DIR, trust_remote_code=False)
        text = " ".join(["alpha", "βeta", "emoji😀", "東京"] * 20)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertGreater(len(token_ids), 32)
        decoder = IncrementalDecoder(tokenizer)

        chunks = [decoder.decode(token_id) for token_id in token_ids]
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.assertEqual("".join(chunks) + decoder.flush(), decoded)
        self.assertLessEqual(len(decoder._token_ids), 32)
        decoder.reset()
        reset_token = tokenizer.encode("Reset", add_special_tokens=False)[0]
        self.assertEqual(
            decoder.decode(reset_token),
            tokenizer.decode(
                [reset_token],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )


class EventAndPublicAPITests(unittest.TestCase):
    def test_event_types_have_stable_payloads(self):
        usage = ChatUsage(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            cached_tokens=1,
        )

        self.assertEqual(ChatTokenEvent("a").type, "token")
        self.assertEqual(ChatFinalTextEvent("done").type, "final_text")
        self.assertEqual(ChatDoneEvent("done", usage).usage.total_tokens, 5)

    def test_chat_session_cannot_be_constructed_or_subclassed_publicly(self):
        with self.assertRaisesRegex(TypeError, "cannot be constructed"):
            ChatSession()
        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):
            type("BadSession", (ChatSession,), {})

    def test_llm_init_validates_store_ids_without_starting_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_llm_bundle(root / "Local" / "model")
            with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                llm = LLM("Local/model")

                self.assertEqual(llm.component_name, "llm")
                self.assertIsNone(llm._active_model_name())

                with self.assertRaisesRegex(ValueError, "search_token_budget"):
                    LLM("Local/model", search_token_budget=0)
                with self.assertRaisesRegex(ValueError, "num_threads"):
                    _make_init_config(
                        model_dir="Local/model",
                        num_threads=-1,
                        lora_dir=None,
                        lora_quant=None,
                        unembed_quant=None,
                    )
        with self.assertRaisesRegex(InvalidRequestError, "model_dir"):
            LLM("Local/does-not-exist")

    def test_unstarted_llm_lifecycle_and_session_guards_are_real(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                write_llm_bundle(root / "Local" / "model")
                with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                    llm = LLM("Local/model", allow_hot_swap=True)
                    await llm.stop()
                    self.assertIsNone(llm._active_model_name())
                    with self.assertRaisesRegex(ComponentLifecycleError, "not running"):
                        llm.open_session()
                    with self.assertRaisesRegex(ComponentLifecycleError, "hot swap"):
                        await llm.swap_model("Local/model")

        import asyncio

        asyncio.run(run())

    def test_start_failure_cleans_runtime_state(self):
        async def run() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                write_llm_bundle(root / "Local" / "model")
                with patch.object(_model_store, "LOCAL_ROOT", root / "Local"):
                    llm = LLM("Local/model")
                    with self.assertRaises(ModelValidationError):
                        await llm.start()
                    self.assertIsNone(llm._runtime)
                    self.assertIsNone(llm._active_model_name())

        import asyncio

        asyncio.run(run())
