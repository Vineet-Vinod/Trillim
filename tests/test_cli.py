"""Tests for the Trillim CLI."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import trillim.cli as cli
from trillim.components.llm._events import ChatDoneEvent, ChatTokenEvent, ChatUsage
from tests.components.llm.support import (
    patched_model_store,
    write_adapter_bundle,
    write_model_bundle,
)


class _FakeStream:
    def __init__(
        self,
        events=None,
        *,
        next_exception: Exception | None = None,
        close_exception: Exception | None = None,
    ) -> None:
        self._events = list(events or [])
        self._next_exception = next_exception
        self._close_exception = close_exception
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_exception is not None:
            exc = self._next_exception
            self._next_exception = None
            raise exc
        if not self._events:
            raise StopIteration
        return self._events.pop(0)

    def close(self) -> None:
        self.closed = True
        if self._close_exception is not None:
            raise self._close_exception


class _FakeSession:
    def __init__(self, *, messages=(), stream=None) -> None:
        self._messages = [dict(message) for message in messages]
        self._stream = stream or _FakeStream()
        self.close_calls = 0

    @property
    def messages(self):
        return tuple(message.copy() for message in self._messages)

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def stream_chat(self):
        return self._stream

    def close(self) -> None:
        self.close_calls += 1


class _FakeRuntimeLLM:
    def __init__(self) -> None:
        self.opened_messages: list[tuple[dict[str, str], ...]] = []
        self.sessions: list[_FakeSession] = []

    def open_session(self, messages=None):
        normalized = tuple(messages or ())
        self.opened_messages.append(normalized)
        session = _FakeSession(messages=normalized)
        self.sessions.append(session)
        return session


class _FakeRuntime:
    def __init__(self, _component) -> None:
        self.llm = _FakeRuntimeLLM()
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class CLITests(unittest.TestCase):
    @contextmanager
    def _patched_model_roots(self):
        with ExitStack() as stack:
            root = stack.enter_context(patched_model_store())
            yield root

    def test_human_size_formats_bytes_and_kibibytes(self):
        self.assertEqual(cli._human_size(10), "10 B")
        self.assertEqual(cli._human_size(1536), "1.5 KB")

    def test_validate_pull_id_only_accepts_trillim_namespace(self):
        self.assertEqual(cli._validate_pull_id("Trillim/demo"), "Trillim/demo")
        with self.assertRaisesRegex(RuntimeError, "only supports Hugging Face IDs"):
            cli._validate_pull_id("Local/demo")

    def test_validate_pull_id_rejects_invalid_shape_and_dot_segments(self):
        for value in ("not-valid", "Trillim/..", "Local/."):
            with self.assertRaisesRegex(RuntimeError, "form Trillim/<name> or Local/<name>"):
                cli._validate_pull_id(value)

    def test_warn_on_trillim_config_ignores_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            with redirect_stdout(output):
                cli._warn_on_trillim_config(Path(temp_dir))
        self.assertEqual(output.getvalue(), "")

    def test_warn_on_trillim_config_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "trillim_config.json").write_text("{", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                cli._warn_on_trillim_config(root)
        self.assertIn("Could not read trillim_config.json", output.getvalue())

    def test_warn_on_trillim_config_ignores_non_object_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "trillim_config.json").write_text("[]", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                cli._warn_on_trillim_config(root)
        self.assertIn("Could not interpret trillim_config.json metadata", output.getvalue())

    def test_warn_on_trillim_config_reports_newer_format_and_platform_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "trillim_config.json").write_text(
                json.dumps({"format_version": 99, "platforms": ["not-this-platform"]}),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("trillim.cli.platform.machine", return_value="arm64"), redirect_stdout(output):
                cli._warn_on_trillim_config(root)
        text = output.getvalue()
        self.assertIn("newer than supported version", text)
        self.assertIn("lists platforms", text)

    def test_warn_on_trillim_config_normalizes_common_platform_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "trillim_config.json").write_text(
                json.dumps({"platforms": ["aarch64"]}),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch("trillim.cli.platform.machine", return_value="arm64"), redirect_stdout(output):
                cli._warn_on_trillim_config(root)
        self.assertEqual(output.getvalue(), "")

    def test_require_remote_code_opt_in_ignores_missing_and_invalid_metadata(self):
        with self._patched_model_roots():
            model = cli._model_store.LOCAL_ROOT / "model"
            model.mkdir(parents=True)
            cli._require_remote_code_opt_in(
                "Local/model",
                label="Model",
                trust_remote_code=False,
            )

            (model / "trillim_config.json").write_text("{", encoding="utf-8")
            cli._require_remote_code_opt_in(
                "Local/model",
                label="Model",
                trust_remote_code=False,
            )

    def test_pull_model_reuses_existing_directory_without_force(self):
        with self._patched_model_roots():
            existing = cli._model_store.DOWNLOADED_ROOT / "demo"
            existing.mkdir(parents=True)
            output = io.StringIO()
            with redirect_stdout(output):
                resolved = cli._pull_model("Trillim/demo", revision=None, force=False)
        self.assertEqual(resolved, existing)
        self.assertIn("already exists", output.getvalue())

    def test_pull_model_downloads_to_downloaded_namespace(self):
        with self._patched_model_roots():
            expected_path = cli._model_store.DOWNLOADED_ROOT / "demo"
            with patch("huggingface_hub.snapshot_download") as download:
                output = io.StringIO()
                with redirect_stdout(output):
                    resolved = cli._pull_model("Trillim/demo", revision="main", force=True)
        self.assertEqual(resolved, expected_path)
        download.assert_called_once_with(
            "Trillim/demo",
            local_dir=str(expected_path),
            revision="main",
            force_download=True,
        )
        self.assertIn("Downloaded to", output.getvalue())

    def test_pull_model_force_removes_existing_directory_before_download(self):
        with self._patched_model_roots():
            expected_path = cli._model_store.DOWNLOADED_ROOT / "demo"
            expected_path.mkdir(parents=True)
            stale_file = expected_path / "stale.bin"
            stale_file.write_bytes(b"stale")

            def fake_download(_repo_id, *, local_dir, revision, force_download):
                self.assertEqual(Path(local_dir), expected_path)
                self.assertFalse(expected_path.exists())
                self.assertEqual(revision, "main")
                self.assertTrue(force_download)

            with patch("huggingface_hub.snapshot_download", side_effect=fake_download):
                cli._pull_model("Trillim/demo", revision="main", force=True)

        self.assertFalse(stale_file.exists())

    def test_pull_model_force_removes_existing_file_before_download(self):
        with self._patched_model_roots():
            expected_path = cli._model_store.DOWNLOADED_ROOT / "demo"
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text("stale", encoding="utf-8")

            def fake_download(_repo_id, *, local_dir, revision, force_download):
                self.assertEqual(Path(local_dir), expected_path)
                self.assertFalse(expected_path.exists())
                self.assertIsNone(revision)
                self.assertTrue(force_download)

            with patch("huggingface_hub.snapshot_download", side_effect=fake_download):
                cli._pull_model("Trillim/demo", revision=None, force=True)

        self.assertFalse(expected_path.exists())

    def test_pull_model_translates_repository_not_found_errors(self):
        missing_error = type("RepositoryNotFoundError", (Exception,), {})("missing")
        with self._patched_model_roots(), patch(
            "huggingface_hub.snapshot_download",
            side_effect=missing_error,
        ):
            with self.assertRaisesRegex(RuntimeError, "not found on Hugging Face"):
                cli._pull_model("Trillim/demo", revision=None, force=True)

    def test_pull_model_translates_gated_repo_errors(self):
        gated_error = type("GatedRepoError", (Exception,), {})("gated")
        with self._patched_model_roots(), patch(
            "huggingface_hub.snapshot_download",
            side_effect=gated_error,
        ):
            with self.assertRaisesRegex(RuntimeError, "hf auth login"):
                cli._pull_model("Trillim/demo", revision=None, force=True)

    def test_pull_model_re_raises_unexpected_download_errors(self):
        with self._patched_model_roots(), patch(
            "huggingface_hub.snapshot_download",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                cli._pull_model("Trillim/demo", revision=None, force=True)

    def test_iter_local_bundles_skips_invalid_entries_and_reports_models_and_adapters(self):
        with self._patched_model_roots():
            downloaded_model = cli._model_store.DOWNLOADED_ROOT / "downloaded-model"
            invalid_model = cli._model_store.DOWNLOADED_ROOT / "invalid-model"
            local_adapter = cli._model_store.LOCAL_ROOT / "local-adapter"
            invalid_adapter = cli._model_store.LOCAL_ROOT / "invalid-adapter"
            invalid = cli._model_store.DOWNLOADED_ROOT / "invalid"
            write_model_bundle(downloaded_model)
            write_model_bundle(invalid_model)
            write_model_bundle(cli._model_store.LOCAL_ROOT / "base-model")
            write_adapter_bundle(
                local_adapter,
                model_root=cli._model_store.LOCAL_ROOT / "base-model",
            )
            (invalid_model / "trillim_config.json").write_text(
                "[]",
                encoding="utf-8",
            )
            invalid_adapter.mkdir(parents=True)
            (invalid_adapter / "qmodel.lora").write_bytes(b"adapter")
            (invalid_adapter / "trillim_config.json").write_text(
                "{}",
                encoding="utf-8",
            )
            invalid.mkdir(parents=True)

            downloaded = cli._iter_local_bundles("Trillim")
            local = cli._iter_local_bundles("Local")

        self.assertEqual([(bundle.model_id, bundle.entry_type) for bundle in downloaded], [("Trillim/downloaded-model", "model")])
        self.assertEqual(
            [(bundle.model_id, bundle.entry_type) for bundle in local],
            [("Local/base-model", "model"), ("Local/local-adapter", "adapter")],
        )

    def test_print_local_table_renders_empty_and_non_empty_sections(self):
        output = io.StringIO()
        with redirect_stdout(output):
            cli._print_local_table("Downloaded", [])
            cli._print_local_table(
                "Local",
                [_LocalBundleLike("Local/demo", "model", "13 B")],
            )
        text = output.getvalue()
        self.assertIn("Downloaded", text)
        self.assertIn("(none)", text)
        self.assertIn("MODEL ID", text)
        self.assertIn("Local/demo", text)

    def test_local_downloaded_ids_returns_empty_set_when_namespace_missing(self):
        with self._patched_model_roots():
            self.assertEqual(cli._local_downloaded_ids(), set())

    def test_list_remote_models_reads_trillim_org_and_marks_local_downloads(self):
        repo_model = SimpleNamespace(
            id="Trillim/model-a",
            siblings=[SimpleNamespace(rfilename="qmodel.tensors")],
            tags=["base_model:meta/base-a"],
            downloads=123,
            last_modified=datetime(2026, 1, 2),
        )
        repo_adapter = SimpleNamespace(
            id="Trillim/adapter-b",
            siblings=[SimpleNamespace(rfilename="qmodel.lora")],
            tags=["base_model:adapter:ignored", "base_model:meta/base-b"],
            downloads=45,
            last_modified=None,
        )
        with self._patched_model_roots():
            write_model_bundle(cli._model_store.DOWNLOADED_ROOT / "model-a")
            with patch("huggingface_hub.list_models", return_value=[repo_model, repo_adapter]):
                entries = cli._list_remote_models()
        self.assertEqual(entries[0]["type"], "model")
        self.assertEqual(entries[0]["base_model"], "meta/base-a")
        self.assertTrue(entries[0]["local"])
        self.assertEqual(entries[1]["type"], "adapter")
        self.assertEqual(entries[1]["base_model"], "meta/base-b")
        self.assertFalse(entries[1]["local"])

    def test_list_remote_models_reports_import_errors(self):
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "huggingface_hub":
                raise ImportError("missing")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "huggingface_hub is required"):
                cli._list_remote_models()

    def test_list_remote_models_translates_hf_http_errors(self):
        http_error = type("HfHubHTTPError", (Exception,), {})("down")
        with patch("huggingface_hub.list_models", side_effect=http_error):
            with self.assertRaisesRegex(RuntimeError, "Failed to fetch models from Hugging Face"):
                cli._list_remote_models()

    def test_print_available_table_renders_empty_and_non_empty_sections(self):
        output = io.StringIO()
        with redirect_stdout(output):
            cli._print_available_table("Models", [])
            cli._print_available_table(
                "Adapters",
                [
                    {
                        "model_id": "Trillim/demo",
                        "base_model": "meta/demo",
                        "downloads": 10,
                        "last_modified": "2026-01-01",
                        "local": True,
                    }
                ],
            )
        text = output.getvalue()
        self.assertIn("Models", text)
        self.assertIn("(none)", text)
        self.assertIn("Adapters", text)
        self.assertIn("STATUS", text)
        self.assertIn("local", text)

    def test_preflight_voice_dependencies_accepts_complete_install(self):
        with patch("trillim.cli.importlib.import_module", return_value=object()) as import_module:
            cli._preflight_voice_dependencies()
        self.assertEqual(import_module.call_count, 4)

    def test_preflight_voice_dependencies_reports_missing_modules(self):
        def load_module(name: str):
            if name in {"numpy", "pocket_tts"}:
                raise ModuleNotFoundError(name)
            return object()

        with patch("trillim.cli.importlib.import_module", side_effect=load_module):
            with self.assertRaisesRegex(RuntimeError, "missing: numpy, pocket_tts"):
                cli._preflight_voice_dependencies()

    def test_stream_assistant_turn_prints_tokens_and_done_text(self):
        session = _FakeSession(
            stream=_FakeStream(
                [
                    ChatTokenEvent("he"),
                    ChatTokenEvent("llo"),
                    ChatDoneEvent("hello", ChatUsage(1, 1, 2, 0)),
                ]
            )
        )
        runtime = SimpleNamespace(llm=SimpleNamespace(open_session=lambda messages: _FakeSession(messages=messages)))
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli._stream_assistant_turn(runtime, session, session.messages)
        self.assertIs(result, session)
        self.assertIn("hello", output.getvalue())

    def test_stream_assistant_turn_reopens_session_after_keyboard_interrupt(self):
        reopened = _FakeSession(messages=({"role": "user", "content": "retry"},))
        runtime = SimpleNamespace(llm=SimpleNamespace(open_session=lambda messages: reopened))
        session = _FakeSession(
            messages=({"role": "user", "content": "retry"},),
            stream=_FakeStream(next_exception=KeyboardInterrupt()),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli._stream_assistant_turn(runtime, session, session.messages)
        self.assertIs(result, reopened)
        self.assertEqual(session.close_calls, 1)
        self.assertIn("Generation cancelled.", output.getvalue())

    def test_stream_assistant_turn_ignores_close_errors(self):
        session = _FakeSession(
            stream=_FakeStream(
                [ChatDoneEvent("done", ChatUsage(1, 1, 2, 0))],
                close_exception=RuntimeError("close failed"),
            )
        )
        runtime = SimpleNamespace(llm=SimpleNamespace(open_session=lambda messages: _FakeSession(messages=messages)))
        output = io.StringIO()
        with redirect_stdout(output):
            cli._stream_assistant_turn(runtime, session, session.messages)
        self.assertIn("done", output.getvalue())

    def test_make_chat_key_bindings_uses_visual_editor_and_updates_buffer(self):
        bindings = cli._make_chat_key_bindings()
        buffer = SimpleNamespace(text="draft", cursor_position=0)
        event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))
        observed: dict[str, object] = {}

        def edit_file(argv: list[str]) -> int:
            observed["editor"] = argv[0]
            temp_path = Path(argv[1])
            observed["path"] = temp_path
            self.assertEqual(temp_path.read_text(encoding="utf-8"), "draft")
            temp_path.write_text("edited text", encoding="utf-8")
            return 0

        with patch.dict(
            "trillim.cli.os.environ",
            {"VISUAL": "nano", "EDITOR": "vim"},
            clear=True,
        ), patch("trillim.cli.subprocess.call", side_effect=edit_file):
            bindings.bindings[0].handler(event)

        self.assertEqual(observed["editor"], "nano")
        self.assertEqual(buffer.text, "edited text")
        self.assertEqual(buffer.cursor_position, len("edited text"))
        self.assertIsInstance(observed["path"], Path)
        self.assertFalse(observed["path"].exists())

    def test_make_chat_key_bindings_uses_editor_when_visual_missing(self):
        bindings = cli._make_chat_key_bindings()
        buffer = SimpleNamespace(text="", cursor_position=0)
        event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))
        called_with: list[str] = []

        def open_editor(argv: list[str]) -> int:
            called_with.append(argv[0])
            return 0

        with patch.dict(
            "trillim.cli.os.environ",
            {"EDITOR": "vim"},
            clear=True,
        ), patch("trillim.cli.subprocess.call", side_effect=open_editor):
            bindings.bindings[0].handler(event)

        self.assertEqual(called_with, ["vim"])

    def test_make_chat_key_bindings_defaults_to_vi(self):
        bindings = cli._make_chat_key_bindings()
        buffer = SimpleNamespace(text="", cursor_position=0)
        event = SimpleNamespace(app=SimpleNamespace(current_buffer=buffer))
        called_with: list[str] = []

        def open_editor(argv: list[str]) -> int:
            called_with.append(argv[0])
            return 0

        with patch.dict(
            "trillim.cli.os.environ",
            {},
            clear=True,
        ), patch("trillim.cli.subprocess.call", side_effect=open_editor):
            bindings.bindings[0].handler(event)

        self.assertEqual(called_with, ["vi"])

    def test_run_chat_supports_reset_prompt_and_quit(self):
        fake_runtime = _FakeRuntime(object())
        with patch("trillim.cli._require_remote_code_opt_in") as require_remote_code, patch(
            "trillim.cli.Runtime",
            return_value=fake_runtime,
        ), patch(
            "trillim.cli.LLM",
            return_value=object(),
        ) as llm_ctor, patch(
            "trillim.cli._make_chat_key_bindings",
            return_value="bindings",
        ), patch(
            "trillim.cli._stream_assistant_turn",
            side_effect=lambda runtime, session, snapshot: session,
        ) as stream_turn, patch(
            "trillim.cli.better_input",
            side_effect=["/new", "hello", "q"],
        ) as prompt_input:
            output = io.StringIO()
            with redirect_stdout(output):
                result = cli._run_chat("Trillim/model", "Local/adapter")
        self.assertEqual(result, 0)
        self.assertEqual(require_remote_code.call_count, 2)
        llm_ctor.assert_called_once_with(
            "Trillim/model",
            lora_dir="Local/adapter",
            trust_remote_code=False,
        )
        self.assertEqual(fake_runtime.llm.opened_messages, [(), ()])
        self.assertEqual(stream_turn.call_count, 1)
        self.assertEqual(
            prompt_input.call_args_list,
            [
                unittest.mock.call("user: ", key_bindings="bindings"),
                unittest.mock.call("user: ", key_bindings="bindings"),
                unittest.mock.call("user: ", key_bindings="bindings"),
            ],
        )
        self.assertIn("Adapter: Local/adapter", output.getvalue())
        self.assertIn("Conversation reset.", output.getvalue())

    def test_run_chat_ignores_blank_prompt(self):
        fake_runtime = _FakeRuntime(object())
        with patch("trillim.cli._require_remote_code_opt_in"), patch(
            "trillim.cli.Runtime",
            return_value=fake_runtime,
        ), patch(
            "trillim.cli.LLM",
            return_value=object(),
        ), patch(
            "trillim.cli._make_chat_key_bindings",
            return_value=object(),
        ), patch(
            "trillim.cli._stream_assistant_turn",
        ) as stream_turn, patch(
            "trillim.cli.better_input",
            side_effect=["   ", "q"],
        ):
            result = cli._run_chat("Trillim/model", None)

        self.assertEqual(result, 0)
        self.assertEqual(fake_runtime.llm.opened_messages, [()])
        stream_turn.assert_not_called()

    def test_run_chat_handles_keyboard_interrupt_and_eof_at_prompt(self):
        fake_runtime = _FakeRuntime(object())
        with patch("trillim.cli._require_remote_code_opt_in"), patch(
            "trillim.cli.Runtime",
            return_value=fake_runtime,
        ), patch(
            "trillim.cli.LLM",
            return_value=object(),
        ), patch(
            "trillim.cli._make_chat_key_bindings",
            return_value=object(),
        ), patch(
            "trillim.cli.better_input",
            side_effect=[KeyboardInterrupt(), EOFError()],
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = cli._run_chat("Trillim/model", None)
        self.assertEqual(result, 0)
        self.assertIn(
            "Commands: /new to reset, q to quit, Ctrl+G for editor",
            output.getvalue(),
        )

    def test_run_chat_requires_trust_remote_code_for_model_and_adapter_metadata(self):
        with self._patched_model_roots(), patch("trillim.cli.LLM") as llm_ctor:
            model = cli._model_store.LOCAL_ROOT / "model"
            adapter = cli._model_store.LOCAL_ROOT / "adapter"
            write_model_bundle(model)
            write_adapter_bundle(adapter, model_root=model)
            model_metadata = json.loads((model / "trillim_config.json").read_text(encoding="utf-8"))
            model_metadata["remote_code"] = True
            (model / "trillim_config.json").write_text(json.dumps(model_metadata), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Model 'Local/model' requires trust_remote_code"):
                cli._run_chat("Local/model", None)

            model_metadata["remote_code"] = False
            (model / "trillim_config.json").write_text(json.dumps(model_metadata), encoding="utf-8")
            adapter_metadata = json.loads((adapter / "trillim_config.json").read_text(encoding="utf-8"))
            adapter_metadata["remote_code"] = True
            (adapter / "trillim_config.json").write_text(json.dumps(adapter_metadata), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Adapter 'Local/adapter' requires trust_remote_code"):
                cli._run_chat("Local/model", "Local/adapter")

        llm_ctor.assert_not_called()

    def test_run_chat_passes_trust_remote_code_to_llm(self):
        fake_runtime = _FakeRuntime(object())
        with patch("trillim.cli.Runtime", return_value=fake_runtime), patch(
            "trillim.cli.LLM",
            return_value=object(),
        ) as llm_ctor, patch(
            "trillim.cli._make_chat_key_bindings",
            return_value=object(),
        ), patch(
            "trillim.cli.better_input",
            side_effect=["q"],
        ):
            result = cli._run_chat("Trillim/model", None, trust_remote_code=True)

        self.assertEqual(result, 0)
        llm_ctor.assert_called_once_with(
            "Trillim/model",
            lora_dir=None,
            trust_remote_code=True,
        )

    def test_run_serve_builds_server_without_voice_by_default(self):
        server = SimpleNamespace(run=lambda **kwargs: kwargs)
        llm = SimpleNamespace()
        with patch("trillim.cli._require_remote_code_opt_in"), patch(
            "trillim.cli.LLM",
            return_value=llm,
        ) as llm_ctor, patch(
            "trillim.cli.Server",
            return_value=server,
        ) as server_ctor:
            result = cli._run_serve("Trillim/model", voice=False)
        self.assertEqual(result, 0)
        llm_ctor.assert_called_once_with("Trillim/model", trust_remote_code=False)
        self.assertEqual(server_ctor.call_args.args, (llm,))
        self.assertEqual(server_ctor.call_args.kwargs["allow_hot_swap"], False)

    def test_run_serve_preflights_voice_dependencies_and_adds_voice_components(self):
        server = SimpleNamespace(run=lambda **kwargs: kwargs)
        llm = SimpleNamespace()
        with patch("trillim.cli._require_remote_code_opt_in"), patch(
            "trillim.cli._preflight_voice_dependencies",
        ) as preflight, patch(
            "trillim.cli.LLM",
            return_value=llm,
        ) as llm_ctor, patch(
            "trillim.cli.STT",
            return_value="stt",
        ) as stt_ctor, patch(
            "trillim.cli.TTS",
            return_value="tts",
        ) as tts_ctor, patch(
            "trillim.cli.Server",
            return_value=server,
        ) as server_ctor:
            result = cli._run_serve("Trillim/model", voice=True)
        self.assertEqual(result, 0)
        preflight.assert_called_once_with()
        llm_ctor.assert_called_once_with("Trillim/model", trust_remote_code=False)
        self.assertEqual(server_ctor.call_args.kwargs["allow_hot_swap"], False)
        stt_ctor.assert_called_once_with()
        tts_ctor.assert_called_once_with()
        self.assertEqual(server_ctor.call_args.args, (llm, "stt", "tts"))

    def test_run_serve_requires_trust_remote_code_for_model_metadata(self):
        with self._patched_model_roots(), patch("trillim.cli.LLM") as llm_ctor:
            model = cli._model_store.LOCAL_ROOT / "model"
            write_model_bundle(model)
            metadata = json.loads((model / "trillim_config.json").read_text(encoding="utf-8"))
            metadata["remote_code"] = True
            (model / "trillim_config.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Model 'Local/model' requires trust_remote_code"):
                cli._run_serve("Local/model", voice=False)

        llm_ctor.assert_not_called()

    def test_run_quantize_command_invokes_quantizer(self):
        args = SimpleNamespace(model_dir="/tmp/model", adapter_dir="/tmp/adapter")
        with patch("trillim.quantize.quantize") as quantize:
            result = cli._run_quantize_command(args)
        self.assertEqual(result, 0)
        quantize.assert_called_once_with("/tmp/model", "/tmp/adapter")

    def test_run_list_command_prints_downloaded_and_local_sections(self):
        with patch("trillim.cli._iter_local_bundles", side_effect=[[], []]), patch(
            "trillim.cli._print_local_table",
        ) as print_table:
            result = cli._run_list_command()
        self.assertEqual(result, 0)
        self.assertEqual(print_table.call_args_list[0].args[0], "Downloaded")
        self.assertEqual(print_table.call_args_list[1].args[0], "Local")

    def test_run_models_command_prints_models_and_adapters_sections(self):
        entries = [
            {"model_id": "Trillim/model", "type": "model"},
            {"model_id": "Trillim/adapter", "type": "adapter"},
        ]
        with patch("trillim.cli._list_remote_models", return_value=entries), patch(
            "trillim.cli._print_available_table",
        ) as print_table:
            result = cli._run_models_command()
        self.assertEqual(result, 0)
        self.assertEqual(print_table.call_args_list[0].args[0], "Models")
        self.assertEqual(print_table.call_args_list[1].args[0], "Adapters")

    def test_build_parser_parses_voice_and_adapter_positionals(self):
        parser = cli.build_parser()
        args = parser.parse_args(["serve", "Trillim/model", "--voice", "--trust-remote-code"])
        self.assertTrue(args.voice)
        self.assertTrue(args.trust_remote_code)
        args = parser.parse_args(["chat", "Trillim/model", "Local/adapter", "--trust-remote-code"])
        self.assertEqual(args.adapter_dir, "Local/adapter")
        self.assertTrue(args.trust_remote_code)
        args = parser.parse_args(["quantize", "/tmp/model", "/tmp/adapter"])
        self.assertEqual(args.model_dir, "/tmp/model")
        self.assertEqual(args.adapter_dir, "/tmp/adapter")

    def test_main_prints_help_when_no_command_is_given(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli.main([])
        self.assertEqual(result, 1)
        self.assertIn("usage:", output.getvalue())

    def test_main_dispatches_each_command_handler(self):
        with patch("trillim.cli._pull_model") as pull_model:
            self.assertEqual(cli.main(["pull", "Trillim/demo"]), 0)
            pull_model.assert_called_once()
        with patch("trillim.cli._run_list_command", return_value=0) as run_list:
            self.assertEqual(cli.main(["list"]), 0)
            run_list.assert_called_once_with()
        with patch("trillim.cli._run_models_command", return_value=0) as run_models:
            self.assertEqual(cli.main(["models"]), 0)
            run_models.assert_called_once_with()
        with patch("trillim.cli._run_chat", return_value=7) as run_chat:
            self.assertEqual(cli.main(["chat", "Trillim/demo"]), 7)
            run_chat.assert_called_once_with(
                "Trillim/demo",
                None,
                trust_remote_code=False,
            )
        with patch("trillim.cli._run_serve", return_value=9) as run_serve:
            self.assertEqual(cli.main(["serve", "Trillim/demo"]), 9)
            run_serve.assert_called_once_with(
                "Trillim/demo",
                voice=False,
                trust_remote_code=False,
            )
        with patch("trillim.cli._run_quantize_command", return_value=1) as run_quantize:
            self.assertEqual(cli.main(["quantize", "/tmp/model"]), 1)
            run_quantize.assert_called_once()

    def test_main_prints_error_for_runtime_failures(self):
        error = io.StringIO()
        with patch("trillim.cli._run_chat", side_effect=RuntimeError("boom")), redirect_stderr(error):
            result = cli.main(["chat", "Trillim/demo"])
        self.assertEqual(result, 1)
        self.assertIn("Error: boom", error.getvalue())


class _LocalBundleLike:
    def __init__(self, model_id: str, entry_type: str, size_human: str) -> None:
        self.model_id = model_id
        self.entry_type = entry_type
        self.size_human = size_human
