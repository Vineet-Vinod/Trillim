from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim import _model_store
from trillim import cli
from trillim._bundle_metadata import CURRENT_FORMAT_VERSION
from tests.support import write_llm_bundle, write_lora_bundle


BONSAI_MODEL_ID = "Trillim/Bonsai-1.7B-TRNQ"
BONSAI_MODEL_DIR = _model_store.store_path_for_id(BONSAI_MODEL_ID)


class CLITests(unittest.TestCase):
    def test_parser_and_main_help_paths(self):
        parser = cli.build_parser()

        self.assertEqual(parser.parse_args(["list"]).command, "list")
        self.assertEqual(parser.parse_args(["chat", "Trillim/model"]).command, "chat")
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = cli.main([])

        self.assertEqual(code, 1)
        self.assertIn("Trillim", stdout.getvalue())

    def test_pull_id_validation_and_platform_normalization(self):
        self.assertEqual(cli._validate_pull_id(" Trillim/BitNet-TRNQ "), "Trillim/BitNet-TRNQ")
        self.assertEqual(cli._normalize_platform_name("ARM64"), "aarch64")
        self.assertEqual(cli._normalize_platform_name("amd64"), "x86_64")
        self.assertEqual(cli._normalize_platform_name("riscv64"), "riscv64")

        with self.assertRaisesRegex(RuntimeError, "only supports"):
            cli._validate_pull_id("Local/model")
        with self.assertRaisesRegex(RuntimeError, "Model IDs"):
            cli._validate_pull_id("bad/model")

    def test_warn_on_trillim_config_reports_bad_and_future_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing"
            missing.mkdir()
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            self.assertEqual(stdout.getvalue(), "")

            (missing / "trillim_config.json").write_text("{", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            self.assertIn("Could not read", stdout.getvalue())

            (missing / "trillim_config.json").write_text("[]", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            self.assertIn("Could not interpret", stdout.getvalue())

            (missing / "trillim_config.json").write_text(
                json.dumps(
                    {
                        "format_version": CURRENT_FORMAT_VERSION + 1,
                        "platforms": ["definitely-not-this-platform"],
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli._warn_on_trillim_config(missing)
            output = stdout.getvalue()
            self.assertIn("newer than supported", output)
            self.assertIn("lists platforms", output)

    def test_remote_code_requires_explicit_trust(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = write_llm_bundle(root / "Trillim" / "remote")
            payload = json.loads((bundle / "trillim_config.json").read_text(encoding="utf-8"))
            payload["remote_code"] = True
            (bundle / "trillim_config.json").write_text(json.dumps(payload), encoding="utf-8")

            with patch.object(_model_store, "DOWNLOADED_ROOT", root / "Trillim"):
                with self.assertRaisesRegex(RuntimeError, "trust_remote_code"):
                    cli._require_remote_code_opt_in(
                        "Trillim/remote",
                        label="Model",
                        trust_remote_code=False,
                    )
                cli._require_remote_code_opt_in(
                    "Trillim/remote",
                    label="Model",
                    trust_remote_code=True,
                )

    def test_local_bundle_listing_and_list_command_use_real_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            downloaded_root = root / "Trillim"
            local_root = root / "Local"
            model_dir = write_llm_bundle(downloaded_root / "model")
            write_lora_bundle(local_root / "adapter", model_dir=model_dir)
            (downloaded_root / "invalid").mkdir(parents=True)

            with (
                patch.object(_model_store, "DOWNLOADED_ROOT", downloaded_root),
                patch.object(_model_store, "LOCAL_ROOT", local_root),
            ):
                downloaded = cli._iter_local_bundles("Trillim")
                local = cli._iter_local_bundles("Local")
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = cli.main(["list"])

        self.assertEqual(code, 0)
        self.assertEqual([bundle.model_id for bundle in downloaded], ["Trillim/model"])
        self.assertEqual([bundle.model_id for bundle in local], ["Local/adapter"])
        output = stdout.getvalue()
        self.assertIn("Downloaded", output)
        self.assertIn("Trillim/model", output)
        self.assertIn("Local/adapter", output)

    def test_downloaded_statuses_marks_stale_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            downloaded_root = root / "Trillim"
            write_llm_bundle(downloaded_root / "valid")
            stale = downloaded_root / "stale"
            stale.mkdir()
            (stale / "trillim_config.json").write_text(
                json.dumps({"format_version": CURRENT_FORMAT_VERSION + 1}),
                encoding="utf-8",
            )
            bad = downloaded_root / "bad-json"
            bad.mkdir()
            (bad / "trillim_config.json").write_text("{", encoding="utf-8")

            with patch.object(_model_store, "DOWNLOADED_ROOT", downloaded_root):
                statuses = cli._downloaded_statuses()

        self.assertEqual(statuses["Trillim/valid"], "local")
        self.assertEqual(statuses["Trillim/stale"], "stale")
        self.assertNotIn("Trillim/bad-json", statuses)

    def test_table_printers_cover_empty_and_populated_output(self):
        bundles = [
            cli._LocalBundle(
                model_id="Trillim/model",
                entry_type="model",
                size_bytes=1024,
                size_human="1 KB",
            )
        ]
        entries = [
            {
                "model_id": "Trillim/model",
                "type": "model",
                "downloads": 7,
                "last_modified": "2026-01-02",
                "base_model": "base",
                "status": "local",
                "local": True,
            }
        ]

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            cli._print_local_table("Empty", [])
            cli._print_local_table("Bundles", bundles)
            cli._print_available_table("Remote", entries)

        output = stdout.getvalue()
        self.assertIn("(none)", output)
        self.assertIn("Trillim/model", output)
        self.assertIn("base", output)

    def test_pull_existing_model_does_not_download_without_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_dir = root / "Trillim" / "model"
            local_dir.mkdir(parents=True)
            with patch.object(_model_store, "DOWNLOADED_ROOT", root / "Trillim"):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    result = cli._pull_model("Trillim/model", revision=None, force=False)

        self.assertEqual(result, local_dir)
        self.assertIn("already exists", stdout.getvalue())

    def test_main_reports_runtime_errors(self):
        with patch.object(cli, "_run_models_command", side_effect=RuntimeError("offline")):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = cli.main(["models"])

        self.assertEqual(code, 1)
        self.assertIn("offline", stderr.getvalue())

    def test_voice_dependency_preflight_passes_with_voice_extra(self):
        cli._preflight_voice_dependencies()

    @unittest.skipUnless(
        BONSAI_MODEL_DIR.is_dir(),
        f"{BONSAI_MODEL_ID} must be installed in the Trillim model store",
    )
    def test_chat_command_starts_real_bonsai_runtime_and_quits(self):
        with patch.object(cli, "better_input", side_effect=["/new", "q"]):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = cli._run_chat(BONSAI_MODEL_ID, None)

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Model: Trillim/Bonsai-1.7B-TRNQ", output)
        self.assertIn("Conversation reset.", output)
