from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from trillim import Runtime, _model_store
import trillim.components.llm.public as llm_public
from trillim.components.llm._config import load_sampling_defaults
from trillim.components.llm._engine import InferenceEngine
from trillim.components.llm._events import ChatDoneEvent, ChatFinalTextEvent, ChatTokenEvent
from trillim.components.llm._model_dir import validate_model_dir
from trillim.components.llm.public import LLM, load_tokenizer
from trillim.errors import ComponentLifecycleError


BONSAI_MODEL_ID = "Trillim/Bonsai-1.7B-TRNQ"
BONSAI_MODEL_DIR = _model_store.store_path_for_id(BONSAI_MODEL_ID)
BITNET_MODEL_ID = "Trillim/BitNet-TRNQ"
BITNET_MODEL_DIR = _model_store.store_path_for_id(BITNET_MODEL_ID)
BITNET_SEARCH_ADAPTER_ID = "Trillim/BitNet-Search-LoRA-TRNQ"
BITNET_SEARCH_ADAPTER_DIR = _model_store.store_path_for_id(BITNET_SEARCH_ADAPTER_ID)


@unittest.skipUnless(
    BONSAI_MODEL_DIR.is_dir(),
    f"{BONSAI_MODEL_ID} must be installed in the Trillim model store",
)
class RealLLMGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_bonsai_inference_engine_starts_generates_and_caches_tokens(self):
        model = validate_model_dir(BONSAI_MODEL_DIR)
        tokenizer = load_tokenizer(BONSAI_MODEL_DIR, trust_remote_code=False)
        engine = InferenceEngine(
            model,
            tokenizer,
            load_sampling_defaults(BONSAI_MODEL_DIR),
            progress_timeout=10.0,
        )
        prompt_tokens = tokenizer.encode("hello", add_special_tokens=True)

        await engine.start()
        try:
            tokens = [
                token
                async for token in engine.generate(
                    prompt_tokens,
                    temperature=0,
                    top_k=1,
                    max_tokens=1,
                )
            ]
            cached_token_count = len(engine._cached_token_ids)
        finally:
            await engine.stop()

        self.assertLessEqual(len(tokens), 1)
        self.assertGreaterEqual(cached_token_count, len(prompt_tokens))

    async def test_bonsai_default_session_streams_events_and_commits_usage(self):
        llm = LLM(BONSAI_MODEL_ID)
        await llm.start()
        try:
            session = llm.open_session()
            events = [
                event
                async for event in session.generate(
                    "hello",
                    temperature=0,
                    top_k=1,
                    max_tokens=1,
                )
            ]
            usage = session._last_usage
            messages = session.messages
            cached_tokens = session.cached_token_count
        finally:
            await llm.stop()

        self.assertTrue(any(isinstance(event, ChatTokenEvent) for event in events))
        self.assertTrue(any(isinstance(event, ChatFinalTextEvent) for event in events))
        self.assertIsInstance(events[-1], ChatDoneEvent)
        self.assertEqual(events[-1].usage, usage)
        self.assertEqual(messages[0], {"role": "user", "content": "hello"})
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIsNotNone(usage)
        self.assertEqual(usage.completion_tokens, 1)
        self.assertEqual(cached_tokens, usage.total_tokens)

    async def test_bonsai_session_public_api_is_bound_to_owner_loop(self):
        llm = LLM(BONSAI_MODEL_ID)
        await llm.start()
        try:
            session = llm.open_session()

            async def read_state_from_thread() -> None:
                _state = session.state

            async def read_messages_from_thread() -> None:
                _messages = session.messages

            async def read_cached_count_from_thread() -> None:
                _cached_token_count = session.cached_token_count

            async def enter_session_from_thread() -> None:
                await session.__aenter__()

            async def exit_session_from_thread() -> None:
                await session.__aexit__(None, None, None)

            async def close_session_from_thread() -> None:
                await session.close()

            async def append_message_from_thread() -> None:
                session.append_message("system", "Be concise.")

            async def generate_from_thread() -> None:
                async for _event in session.generate("hello", max_tokens=1):
                    pass

            async def collect_from_thread() -> None:
                await session.collect("hello", max_tokens=1)

            async def new_chat_from_thread() -> None:
                session.new_chat()

            async def run_on_new_loop(operation) -> None:
                def run() -> None:
                    asyncio.run(operation())

                await asyncio.to_thread(run)

            for operation in (
                read_state_from_thread,
                read_messages_from_thread,
                read_cached_count_from_thread,
                enter_session_from_thread,
                exit_session_from_thread,
                close_session_from_thread,
                append_message_from_thread,
                generate_from_thread,
                collect_from_thread,
                new_chat_from_thread,
            ):
                with self.subTest(operation=operation.__name__):
                    with self.assertRaisesRegex(ComponentLifecycleError, "one event loop"):
                        await run_on_new_loop(operation)
        finally:
            await llm.stop()

    async def test_bonsai_search_harness_no_tool_path_uses_generation_lane(self):
        llm = LLM(BONSAI_MODEL_ID, harness_name="search")
        await llm.start()
        try:
            session = llm.open_session()
            text = await session.collect(
                "hello",
                temperature=0,
                top_k=1,
                max_tokens=1,
            )
            usage = session._last_usage
            messages = session.messages
        finally:
            await llm.stop()

        self.assertIsInstance(text, str)
        self.assertEqual(messages[-1], {"role": "assistant", "content": text})
        self.assertIsNotNone(usage)
        self.assertEqual(usage.completion_tokens, 1)

    def test_bonsai_runtime_sync_proxy_collects_with_llm_component(self):
        runtime = Runtime(LLM(BONSAI_MODEL_ID))
        with runtime:
            session = runtime.llm.open_session()
            text = session.collect(
                "hello",
                temperature=0,
                top_k=1,
                max_tokens=1,
            )

        self.assertIsInstance(text, str)

    def test_bonsai_runtime_sync_proxy_streams_session_iterator(self):
        runtime = Runtime(LLM(BONSAI_MODEL_ID))
        with runtime:
            with runtime.llm.open_session() as session:
                events = list(
                    session.generate(
                        "hello",
                        temperature=0,
                        top_k=1,
                        max_tokens=1,
                    )
                )

        self.assertTrue(any(isinstance(event, ChatTokenEvent) for event in events))
        self.assertIsInstance(events[-1], ChatDoneEvent)


@unittest.skipUnless(
    BITNET_MODEL_DIR.is_dir() and BITNET_SEARCH_ADAPTER_DIR.is_dir(),
    f"{BITNET_MODEL_ID} and {BITNET_SEARCH_ADAPTER_ID} must be installed",
)
class RealSearchHarnessGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_bitnet_search_adapter_runs_real_search_harness(self):
        with patch.object(llm_public, "TOKEN_PROGRESS_TIMEOUT_SECONDS", 30.0):
            llm = LLM(
                BITNET_MODEL_ID,
                lora_dir=BITNET_SEARCH_ADAPTER_ID,
                num_threads=4,
                harness_name="search",
                search_provider="ddgs",
                search_token_budget=32,
            )
            await llm.start()
            try:
                session = llm.open_session()
                events = [
                    event
                    async for event in session.generate(
                        "Search the web for current OpenAI news, then answer in one short sentence.",
                        temperature=0,
                        top_k=1,
                        max_tokens=32,
                    )
                ]
                messages = session.messages
                usage = session._last_usage
            finally:
                await llm.stop()

        self.assertTrue(any(message["role"] == "search" for message in messages))
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertTrue(any(isinstance(event, ChatTokenEvent) for event in events))
        self.assertIsInstance(events[-1], ChatDoneEvent)
        self.assertIsNotNone(usage)
        self.assertGreater(usage.prompt_tokens, 0)
        self.assertGreater(usage.completion_tokens, 0)

    async def test_bitnet_search_adapter_surfaces_real_brave_authentication_error(self):
        with patch.object(llm_public, "TOKEN_PROGRESS_TIMEOUT_SECONDS", 30.0):
            llm = LLM(
                BITNET_MODEL_ID,
                lora_dir=BITNET_SEARCH_ADAPTER_ID,
                num_threads=4,
                harness_name="search",
                search_provider="brave",
                search_token_budget=32,
            )
            await llm.start()
            try:
                session = llm.open_session()
                with self.assertRaisesRegex(RuntimeError, "SEARCH_API_KEY"):
                    await session.collect(
                        "Search the web for current OpenAI news, then answer in one short sentence.",
                        temperature=0,
                        top_k=1,
                        max_tokens=32,
                    )
                self.assertEqual(session.state, "idle")
            finally:
                await llm.stop()
