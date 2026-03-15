# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the interactive CLI chat helpers."""

import asyncio
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from trillim import ChatSession
from trillim.events import ChatSearchResultEvent, ChatSearchStartedEvent, ChatTokenEvent
import trillim.inference as inference


class _Tokenizer:
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = True):
        return list(range(len(text)))


class _Engine:
    def __init__(self, *, max_context_tokens: int):
        self.model_dir = "models/fake"
        self.arch_config = SimpleNamespace(max_position_embeddings=max_context_tokens)
        self.tokenizer = _Tokenizer()
        self.reset_calls = 0

    def reset_prompt_cache(self) -> None:
        self.reset_calls += 1


class _Harness:
    def __init__(self, *, max_context_tokens: int):
        self.engine = _Engine(max_context_tokens=max_context_tokens)
        self.arch_config = self.engine.arch_config


class _LLM:
    def __init__(self, *, max_context_tokens: int):
        self.model_name = "fake"
        self.engine = _Engine(max_context_tokens=max_context_tokens)
        self.harness = _Harness(max_context_tokens=max_context_tokens)
        self.harness.engine = self.engine
        self._session_generation = 0

    def _require_started(self):
        return self.engine, self.harness

    def _chat_sampling(self, **sampling):
        return sampling

    @property
    def max_context_tokens(self) -> int:
        return self.engine.arch_config.max_position_embeddings

    def session(self, messages: list[dict] | None = None):
        copied = [{"role": m["role"], "content": m["content"]} for m in messages or []]
        return ChatSession(self, copied)


class _SessionStub:
    def __init__(self):
        self.sampling = None

    async def stream_chat(self, **sampling):
        self.sampling = sampling
        yield ChatSearchStartedEvent(query="cats")
        yield ChatSearchResultEvent(
            query="cats",
            content="curated result",
            available=True,
        )
        yield ChatSearchResultEvent(
            query="cats",
            content="Search unavailable",
            available=False,
        )
        yield ChatTokenEvent(text="O")
        yield ChatTokenEvent(text="K")


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _MainEngineStub:
    def __init__(self):
        self.default_params = {"temperature": 0.4}


class _MainLLMStub:
    instances: list["_MainLLMStub"] = []

    def __init__(self, model_dir, **kwargs):
        self.model_dir = model_dir
        self.kwargs = kwargs
        self.engine = _MainEngineStub()
        self.start_calls = 0
        self.stop_calls = 0
        self._search_provider = None
        _MainLLMStub.instances.append(self)

    async def start(self):
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1


class InferenceLoopTests(unittest.TestCase):
    def test_run_chat_loop_uses_chat_sessions_and_filters_sampling_params(self):
        loop = asyncio.new_event_loop()
        llm = _LLM(max_context_tokens=128)
        streamed: list[tuple[type, tuple[dict, ...], dict]] = []

        async def fake_stream_response(chat, sampling_params):
            streamed.append((type(chat), chat.messages, dict(sampling_params)))

        try:
            with (
                patch("trillim.inference._make_key_bindings", return_value=object()),
                patch(
                    "trillim.inference.better_input",
                    side_effect=["hello", "/new", "again", "q"],
                ),
                patch("trillim.inference._stream_response", new=fake_stream_response),
                patch("builtins.print"),
            ):
                inference._run_chat_loop(
                    loop,
                    llm,
                    {
                        "temperature": 0.6,
                        "top_k": 50,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                        "rep_penalty_lookback": 64,
                    },
                )
        finally:
            loop.close()

        self.assertEqual(
            streamed,
            [
                (
                    ChatSession,
                    ({"role": "user", "content": "hello"},),
                    {
                        "temperature": 0.6,
                        "top_k": 50,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                    },
                ),
                (
                    ChatSession,
                    ({"role": "user", "content": "again"},),
                    {
                        "temperature": 0.6,
                        "top_k": 50,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                    },
                ),
            ],
        )
        self.assertEqual(llm.engine.reset_calls, 1)

    def test_run_chat_loop_resets_to_latest_message_and_skips_oversized_input(self):
        loop = asyncio.new_event_loop()
        llm = _LLM(max_context_tokens=30)
        streamed: list[tuple[dict, ...]] = []

        async def fake_stream_response(chat, sampling_params):
            streamed.append(chat.messages)
            chat._append_message("assistant", "ok")

        try:
            with (
                patch("trillim.inference._make_key_bindings", return_value=object()),
                patch(
                    "trillim.inference.better_input",
                    side_effect=[
                        "hi",
                        "yo",
                        "this message is definitely too long",
                        "q",
                    ],
                ),
                patch("trillim.inference._stream_response", new=fake_stream_response),
                patch("builtins.print") as print_mock,
            ):
                inference._run_chat_loop(loop, llm, {})
        finally:
            loop.close()

        self.assertEqual(
            streamed,
            [
                ({"role": "user", "content": "hi"},),
                ({"role": "user", "content": "yo"},),
            ],
        )
        self.assertEqual(llm.engine.reset_calls, 2)
        printed = [call.args[0] for call in print_mock.call_args_list if call.args]
        self.assertEqual(
            printed.count("Context window full (30 tokens). Starting new conversation."),
            2,
        )
        self.assertIn(
            "Last message exceeds the context window (30 tokens). Shorten it and try again.",
            printed,
        )


class InferenceStreamingTests(unittest.TestCase):
    def test_stream_response_prints_status_markers_and_tokens(self):
        session = _SessionStub()

        with patch("builtins.print") as print_mock:
            asyncio.run(
                inference._stream_response(
                    session,
                    {"temperature": 0.4, "max_tokens": 32},
                )
            )

        self.assertEqual(session.sampling, {"temperature": 0.4, "max_tokens": 32})
        self.assertEqual(
            print_mock.call_args_list,
            [
                unittest.mock.call("[Searching: cats]", flush=True),
                unittest.mock.call("[Synthesizing...]", flush=True),
                unittest.mock.call("[Search unavailable]", flush=True),
                unittest.mock.call("O", end="", flush=True),
                unittest.mock.call("K", end="", flush=True),
            ],
        )


class InferenceBootstrapTests(unittest.TestCase):
    def test_make_key_bindings_opens_editor_and_replaces_buffer_contents(self):
        kb = inference._make_key_bindings()
        handler = kb.bindings[0].handler
        buffer = SimpleNamespace(text="draft", cursor_position=0)
        event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))

        def fake_call(argv):
            with open(argv[1], "w", encoding="utf-8") as handle:
                handle.write("edited in editor")
            return 0

        with (
            patch.dict("os.environ", {"EDITOR": "nano"}, clear=False),
            patch("trillim.inference.subprocess.call", side_effect=fake_call) as call_mock,
        ):
            handler(event)

        self.assertEqual(buffer.text, "edited in editor")
        self.assertEqual(buffer.cursor_position, len("edited in editor"))
        self.assertEqual(call_mock.call_args.args[0][0], "nano")

    def test_main_builds_llm_and_runs_chat_loop_with_parsed_flags(self):
        _MainLLMStub.instances.clear()
        fake_server_module = _module("trillim.server", LLM=_MainLLMStub)

        with (
            patch.object(
                sys,
                "argv",
                [
                    "trillim",
                    "models/demo/",
                    "--lora",
                    "adapter/demo",
                    "--threads",
                    "8",
                    "--lora-quant",
                    "q4",
                    "--unembed-quant",
                    "q8",
                    "--trust-remote-code",
                    "--harness",
                    "search",
                    "--search-provider",
                    "brave",
                ],
            ),
            patch.dict("sys.modules", {"trillim.server": fake_server_module}),
            patch("trillim.inference._run_chat_loop") as loop_mock,
        ):
            inference.main()

        llm = _MainLLMStub.instances[-1]
        self.assertEqual(llm.model_dir, "models/demo")
        self.assertEqual(
            llm.kwargs,
            {
                "adapter_dir": "adapter/demo",
                "num_threads": 8,
                "trust_remote_code": True,
                "lora_quant": "q4",
                "unembed_quant": "q8",
                "harness_name": "search",
            },
        )
        self.assertEqual(llm._search_provider, "brave")
        self.assertEqual(llm.start_calls, 1)
        self.assertEqual(llm.stop_calls, 1)
        self.assertEqual(loop_mock.call_args.args[1], llm)
        self.assertEqual(loop_mock.call_args.args[2], {"temperature": 0.4})

    def test_main_requires_a_model_argument(self):
        with patch.object(sys, "argv", ["trillim"]):
            with self.assertRaisesRegex(ValueError, "Usage: trillim chat <model_directory>"):
                inference.main()

    def test_main_requires_lora_path_when_flag_is_present(self):
        with patch.object(sys, "argv", ["trillim", "models/demo", "--lora"]):
            with self.assertRaisesRegex(ValueError, "--lora requires an adapter directory path"):
                inference.main()

    def test_main_prints_broken_pipe_and_generic_errors(self):
        _MainLLMStub.instances.clear()
        fake_server_module = _module("trillim.server", LLM=_MainLLMStub)

        with (
            patch.object(sys, "argv", ["trillim", "models/demo"]),
            patch.dict("sys.modules", {"trillim.server": fake_server_module}),
            patch("trillim.inference._run_chat_loop", side_effect=BrokenPipeError()),
            patch("builtins.print") as print_mock,
        ):
            inference.main()

        self.assertEqual(
            [call.args[0] for call in print_mock.call_args_list],
            [
                "\nError: Inference engine crashed.",
                "\nIf you think the engine is broken, please report the bug!",
            ],
        )

        with (
            patch.object(sys, "argv", ["trillim", "models/demo"]),
            patch.dict("sys.modules", {"trillim.server": fake_server_module}),
            patch("trillim.inference._run_chat_loop", side_effect=RuntimeError("boom")),
            patch("builtins.print") as print_mock,
        ):
            inference.main()

        self.assertEqual(print_mock.call_args.args[0], "\nAn error occurred: boom")

    def test_run_chat_loop_treats_eof_as_quit(self):
        loop = asyncio.new_event_loop()
        llm = _LLM(max_context_tokens=64)
        try:
            with (
                patch("trillim.inference._make_key_bindings", return_value=object()),
                patch("trillim.inference.better_input", side_effect=EOFError()),
                patch("trillim.inference._stream_response") as stream_mock,
                patch("builtins.print") as print_mock,
            ):
                inference._run_chat_loop(loop, llm, {})
        finally:
            loop.close()

        stream_mock.assert_not_called()
        self.assertTrue(print_mock.call_args.args[0].startswith("Talk to fake"))

    def test_inference_module_runs_main_when_executed_as_script(self):
        _MainLLMStub.instances.clear()
        with (
            patch.object(sys, "argv", ["trillim", "models/demo"]),
            patch.dict("sys.modules", {"trillim.server": _module("trillim.server", LLM=_MainLLMStub)}),
            patch("prompt_toolkit.prompt", return_value="q"),
            patch("builtins.print"),
        ):
            runpy.run_path(inference.__file__, run_name="__main__")

        self.assertEqual(_MainLLMStub.instances[-1].start_calls, 1)
        self.assertEqual(_MainLLMStub.instances[-1].stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
