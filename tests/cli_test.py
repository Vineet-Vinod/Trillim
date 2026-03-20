# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for the unified Trillim CLI entry point."""

from __future__ import annotations

import io
import json
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import trillim.cli as cli


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FakeLLM:
    instances: list["_FakeLLM"] = []

    def __init__(self, model_dir, **kwargs):
        self.model_dir = model_dir
        self.kwargs = kwargs
        _FakeLLM.instances.append(self)


class _FakeWhisper:
    def __init__(self, *, model_size):
        self.model_size = model_size


class _FakeTTS:
    def __init__(self, *, voices_dir):
        self.voices_dir = voices_dir


class _FakeServer:
    last_instance: "_FakeServer | None" = None

    def __init__(self, *components):
        self.components = components
        self.run_calls: list[tuple[tuple, dict]] = []
        _FakeServer.last_instance = self

    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))


class CliTests(unittest.TestCase):
    def test_resolve_updates_model_dir_when_present(self):
        args = SimpleNamespace(model_dir="Org/Model")
        model_store_module = _module("trillim.model_store", resolve_model_dir=lambda value: f"/resolved/{value}")

        with patch.dict("sys.modules", {"trillim.model_store": model_store_module}):
            cli._resolve(args)

        self.assertEqual(args.model_dir, "/resolved/Org/Model")

    def test_resolve_ignores_namespaces_without_model_dir(self):
        args = SimpleNamespace(other="value")
        cli._resolve(args)
        self.assertEqual(args.other, "value")

    def test_cmd_quantize_invokes_quantize_main_with_expected_argv(self):
        seen_argv: list[str] = []

        def quantize_main():
            seen_argv[:] = sys.argv

        args = SimpleNamespace(
            model_dir="Org/Model",
            model=True,
            adapter="/adapter",
            language_model_only=True,
        )

        with (
            patch("trillim.cli._resolve", side_effect=lambda ns: setattr(ns, "model_dir", "/resolved/model")),
            patch.object(sys, "argv", ["trillim"]),
            patch.dict("sys.modules", {"trillim.quantize": _module("trillim.quantize", main=quantize_main)}),
        ):
            cli._cmd_quantize(args)

        self.assertEqual(
            seen_argv,
            [
                "trillim",
                "/resolved/model",
                "--model",
                "--adapter",
                "/adapter",
                "--language-model-only",
            ],
        )

    def test_cmd_quantize_exits_on_errors(self):
        args = SimpleNamespace(
            model_dir="Org/Model",
            model=False,
            adapter=None,
            language_model_only=False,
        )
        stderr = io.StringIO()

        with (
            patch("trillim.cli._resolve", side_effect=RuntimeError("broken")),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli._cmd_quantize(args)

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Error: broken", stderr.getvalue())

    def test_cmd_chat_builds_rich_argv_and_invokes_inference_main(self):
        seen_argv: list[str] = []

        def inference_main():
            seen_argv[:] = sys.argv

        resolved_values = {"Org/Model": "/resolved/model", "Org/Adapter": "/resolved/adapter"}
        args = SimpleNamespace(
            model_dir="Org/Model",
            lora="Org/Adapter",
            threads=8,
            lora_quant="q4",
            unembed_quant="q8",
            trust_remote_code=True,
            harness="search",
            search_provider="brave",
        )

        with (
            patch("trillim.cli._resolve", side_effect=lambda ns: setattr(ns, "model_dir", resolved_values[ns.model_dir])),
            patch.object(sys, "argv", ["trillim"]),
            patch.dict(
                "sys.modules",
                {
                    "trillim.model_store": _module(
                        "trillim.model_store",
                        resolve_model_dir=lambda value: resolved_values[value],
                    ),
                    "trillim.inference": _module("trillim.inference", main=inference_main),
                },
            ),
        ):
            cli._cmd_chat(args)

        self.assertEqual(
            seen_argv,
            [
                "trillim",
                "/resolved/model",
                "--lora",
                "/resolved/adapter",
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
        )

    def test_cmd_chat_omits_default_optional_flags(self):
        seen_argv: list[str] = []

        def inference_main():
            seen_argv[:] = sys.argv

        args = SimpleNamespace(
            model_dir="Org/Model",
            lora=None,
            threads=0,
            lora_quant=None,
            unembed_quant=None,
            trust_remote_code=False,
            harness="default",
            search_provider="ddgs",
        )

        with (
            patch("trillim.cli._resolve", side_effect=lambda ns: setattr(ns, "model_dir", "/resolved/model")),
            patch.object(sys, "argv", ["trillim"]),
            patch.dict("sys.modules", {"trillim.inference": _module("trillim.inference", main=inference_main)}),
        ):
            cli._cmd_chat(args)

        self.assertEqual(seen_argv, ["trillim", "/resolved/model"])

    def test_cmd_chat_exits_on_errors(self):
        args = SimpleNamespace(
            model_dir="Org/Model",
            lora=None,
            threads=0,
            lora_quant=None,
            unembed_quant=None,
            trust_remote_code=False,
            harness="default",
            search_provider="ddgs",
        )
        stderr = io.StringIO()

        with (
            patch("trillim.cli._resolve", side_effect=FileNotFoundError("missing")),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli._cmd_chat(args)

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Error: missing", stderr.getvalue())

    def test_cmd_serve_runs_server_with_voice_components(self):
        _FakeLLM.instances.clear()
        _FakeServer.last_instance = None
        args = SimpleNamespace(
            model_dir="Org/Model",
            host="0.0.0.0",
            port=9000,
            voice=True,
            whisper_model="tiny.en",
            voices_dir="/voices",
            threads=3,
            lora_quant="q4",
            unembed_quant="q8",
            trust_remote_code=True,
        )

        with (
            patch("trillim.cli._resolve", side_effect=lambda ns: setattr(ns, "model_dir", "/resolved/model")),
            patch.dict(
                "sys.modules",
                {
                    "trillim": _module(
                        "trillim",
                        LLM=_FakeLLM,
                        Whisper=_FakeWhisper,
                        TTS=_FakeTTS,
                        Server=_FakeServer,
                    )
                },
            ),
        ):
            cli._cmd_serve(args)

        llm = _FakeLLM.instances[-1]
        self.assertEqual(llm.model_dir, "/resolved/model")
        self.assertEqual(
            llm.kwargs,
            {
                "num_threads": 3,
                "trust_remote_code": True,
                "lora_quant": "q4",
                "unembed_quant": "q8",
            },
        )
        self.assertEqual(len(_FakeServer.last_instance.components), 3)
        self.assertEqual(_FakeServer.last_instance.run_calls, [((), {"host": "0.0.0.0", "port": 9000})])
        self.assertEqual(cli.os.environ["TOKENIZERS_PARALLELISM"], "false")

    def test_cmd_serve_exits_on_runtime_errors(self):
        stderr = io.StringIO()
        args = SimpleNamespace(
            model_dir="Org/Model",
            host="127.0.0.1",
            port=8000,
            voice=False,
            whisper_model="base.en",
            voices_dir="/voices",
            threads=0,
            lora_quant=None,
            unembed_quant=None,
            trust_remote_code=False,
        )

        with (
            patch("trillim.cli._resolve", side_effect=lambda ns: setattr(ns, "model_dir", "/resolved/model")),
            patch(
                "sys.stderr",
                stderr,
            ),
            patch.dict(
                "sys.modules",
                {
                    "trillim": _module(
                        "trillim",
                        LLM=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
                        Whisper=_FakeWhisper,
                        TTS=_FakeTTS,
                        Server=_FakeServer,
                    )
                },
            ),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli._cmd_serve(args)

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Error: boom", stderr.getvalue())

    def test_cmd_pull_invokes_model_store_and_maps_runtime_errors(self):
        pull_model = Mock()
        args = SimpleNamespace(model_id="Org/Model", revision="main", force=True)

        with patch.dict("sys.modules", {"trillim.model_store": _module("trillim.model_store", pull_model=pull_model)}):
            cli._cmd_pull(args)
        pull_model.assert_called_once_with("Org/Model", revision="main", force=True)

        stderr = io.StringIO()
        with (
            patch.dict(
                "sys.modules",
                {"trillim.model_store": _module("trillim.model_store", pull_model=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))},
            ),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli._cmd_pull(args)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Error: offline", stderr.getvalue())

    def test_cmd_models_supports_json_empty_and_table_output(self):
        json_stdout = io.StringIO()
        with (
            patch.dict(
                "sys.modules",
                {"trillim.model_store": _module("trillim.model_store", list_models=lambda: [{"model_id": "m"}], list_adapters=lambda: [{"model_id": "a"}])},
            ),
            patch("sys.stdout", json_stdout),
        ):
            cli._cmd_models(SimpleNamespace(json=True))
        self.assertEqual(json.loads(json_stdout.getvalue()), {"models": [{"model_id": "m"}], "adapters": [{"model_id": "a"}]})

        empty_stdout = io.StringIO()
        with (
            patch.dict(
                "sys.modules",
                {"trillim.model_store": _module("trillim.model_store", list_models=lambda: [], list_adapters=lambda: [])},
            ),
            patch("sys.stdout", empty_stdout),
        ):
            cli._cmd_models(SimpleNamespace(json=False))
        self.assertIn("No models found. Run: trillim pull <org/model>", empty_stdout.getvalue())

        table_stdout = io.StringIO()
        models = [
            {
                "model_id": "Org/Base",
                "architecture": "llama",
                "size_human": "2 KB",
                "source_model": "meta/base",
                "base_model_config_hash": "hash-1",
            },
            {
                "model_id": "Org/Base2",
                "architecture": "llama",
                "size_human": "4 KB",
                "source_model": "meta/base2",
                "base_model_config_hash": "hash-1",
            },
        ]
        adapters = [{"model_id": "Org/Adapter", "size_human": "1 KB", "base_model_config_hash": "hash-1"}]
        with (
            patch.dict(
                "sys.modules",
                {"trillim.model_store": _module("trillim.model_store", list_models=lambda: models, list_adapters=lambda: adapters)},
            ),
            patch("sys.stdout", table_stdout),
        ):
            cli._cmd_models(SimpleNamespace(json=False))
        output = table_stdout.getvalue()
        self.assertIn("Models", output)
        self.assertIn("Adapters", output)
        self.assertIn("Org/Base2", output)

    def test_cmd_list_supports_json_empty_table_and_error_output(self):
        entries = [
            {"model_id": "Org/Base", "type": "model", "downloads": 10, "last_modified": "2026-03-14", "base_model": "meta/base", "local": True},
            {"model_id": "Org/Adapter", "type": "adapter", "downloads": 2, "last_modified": "", "base_model": "", "local": False},
        ]

        json_stdout = io.StringIO()
        with (
            patch.dict("sys.modules", {"trillim.model_store": _module("trillim.model_store", list_available_models=lambda: entries)}),
            patch("sys.stdout", json_stdout),
        ):
            cli._cmd_list(SimpleNamespace(json=True))
        self.assertEqual(json.loads(json_stdout.getvalue()), entries)

        table_stdout = io.StringIO()
        with (
            patch.dict("sys.modules", {"trillim.model_store": _module("trillim.model_store", list_available_models=lambda: entries)}),
            patch("sys.stdout", table_stdout),
        ):
            cli._cmd_list(SimpleNamespace(json=False))
        self.assertIn("Models", table_stdout.getvalue())
        self.assertIn("Adapters", table_stdout.getvalue())
        self.assertIn("local", table_stdout.getvalue())

        empty_stdout = io.StringIO()
        with (
            patch.dict("sys.modules", {"trillim.model_store": _module("trillim.model_store", list_available_models=lambda: [])}),
            patch("sys.stdout", empty_stdout),
        ):
            cli._cmd_list(SimpleNamespace(json=False))
        self.assertIn("No models found in the Trillim organization.", empty_stdout.getvalue())

        stderr = io.StringIO()
        with (
            patch.dict(
                "sys.modules",
                {"trillim.model_store": _module("trillim.model_store", list_available_models=lambda: (_ for _ in ()).throw(RuntimeError("offline")))},
            ),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli._cmd_list(SimpleNamespace(json=False))
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Error: offline", stderr.getvalue())

    def test_print_available_table_ignores_empty_entries(self):
        with patch("builtins.print") as print_mock:
            cli._print_available_table("Models", [])
        print_mock.assert_not_called()

    def test_main_prints_help_when_no_subcommand_is_provided(self):
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", ["trillim"]),
            patch("sys.stdout", stdout),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli.main()

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("usage: trillim", stdout.getvalue())

    def test_main_dispatches_to_selected_subcommand(self):
        with (
            patch.object(sys, "argv", ["trillim", "models", "--json"]),
            patch("trillim.cli._cmd_models") as cmd_models,
        ):
            cli.main()

        cmd_models.assert_called_once()
        self.assertTrue(cmd_models.call_args.args[0].json)

    def test_cli_module_runs_main_when_executed_as_script(self):
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", ["trillim", "models", "--json"]),
            patch.dict(
                "sys.modules",
                {
                    "trillim.model_store": _module(
                        "trillim.model_store",
                        list_models=lambda: [],
                        list_adapters=lambda: [],
                    )
                },
            ),
            patch("sys.stdout", stdout),
        ):
            runpy.run_path(cli.__file__, run_name="__main__")

        self.assertEqual(json.loads(stdout.getvalue()), {"models": [], "adapters": []})


if __name__ == "__main__":
    unittest.main()
