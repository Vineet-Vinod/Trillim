from __future__ import annotations

import asyncio
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from trillim import LLM, Runtime, Server, STT, TTS
from trillim import _model_store
from trillim._app import build_app
from trillim._bundle_metadata import canonicalize_model_config, compute_base_model_config_hash
from trillim.components import Component
from trillim.errors import ContextOverflowError, OperationCancelledError
from trillim.utils.cancellation import CancellationSource
from trillim.utils.filesystem import atomic_write_bytes, ensure_within_root, unlink_if_exists
from trillim.utils.formatting import human_size
from trillim.utils.ids import stable_id


class PackageAndErrorTests(unittest.TestCase):
    def test_public_exports_are_available(self):
        self.assertIs(LLM.__name__, "LLM")
        self.assertIs(STT.__name__, "STT")
        self.assertIs(TTS.__name__, "TTS")
        self.assertIs(Runtime.__name__, "Runtime")
        self.assertIs(Server.__name__, "Server")

    def test_context_overflow_error_exposes_counts(self):
        error = ContextOverflowError(10, 5)

        self.assertEqual(error.token_count, 10)
        self.assertEqual(error.limit, 5)
        self.assertIn("exceeds", str(error))


class ModelStoreTests(unittest.TestCase):
    def test_store_id_parsing_and_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(_model_store, "DOWNLOADED_ROOT", root / "Trillim"), patch.object(
                _model_store, "LOCAL_ROOT", root / "Local"
            ):
                path = root / "Local" / "model"
                path.mkdir(parents=True)

                self.assertEqual(_model_store.parse_store_id(" Local/model "), ("Local", "model"))
                self.assertEqual(_model_store.store_path_for_id("Local/model"), path)
                self.assertEqual(_model_store.resolve_existing_store_id("Local/model"), path)

                with self.assertRaisesRegex(ValueError, "Model IDs"):
                    _model_store.parse_store_id("bad/model")
                with self.assertRaisesRegex(ValueError, "not found"):
                    _model_store.resolve_existing_store_id("Trillim/missing")


class UtilityTests(unittest.TestCase):
    def test_bundle_metadata_hash_uses_canonical_config_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            payload = {
                "architectures": ["Wrapper"],
                "text_config": {
                    "architectures": ["Inner"],
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "vocab_size": 100,
                },
            }
            (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")

            canonical = canonicalize_model_config(payload)
            digest = compute_base_model_config_hash(model_dir)

        self.assertEqual(canonical["architectures"], ["Wrapper"])
        self.assertEqual(canonical["hidden_size"], 128)
        self.assertEqual(len(digest), 64)

    def test_bundle_metadata_rejects_invalid_hash_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                compute_base_model_config_hash(model_dir)

            (model_dir / "config.json").write_text(
                json.dumps({"num_attention_heads": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "positive integer"):
                compute_base_model_config_hash(model_dir)

    def test_human_size_formats_base_1024_units(self):
        self.assertEqual(human_size(0), "0 B")
        self.assertEqual(human_size(1024), "1 KB")
        self.assertEqual(human_size(1536), "1.5 KB")
        self.assertEqual(human_size(1024**5), "1.0 PB")

    def test_atomic_write_and_root_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = atomic_write_bytes(root / "nested" / "file.bin", b"data")

            self.assertEqual(target.read_bytes(), b"data")
            self.assertEqual(oct(target.stat().st_mode & 0o777), "0o600")
            self.assertEqual(ensure_within_root(target, root), target.resolve())
            with self.assertRaisesRegex(ValueError, "outside allowed root"):
                ensure_within_root(root.parent / "outside", root)
            unlink_if_exists(target)
            self.assertFalse(target.exists())

    def test_stable_id_validates_prefix_and_digest_size(self):
        self.assertEqual(stable_id("item", "value"), stable_id("item", b"value"))
        with self.assertRaisesRegex(ValueError, "prefix"):
            stable_id("bad-prefix", "value")
        with self.assertRaisesRegex(ValueError, "digest_size"):
            stable_id("item", "value", digest_size=3)

    def test_cancellation_source_and_token(self):
        source = CancellationSource()
        self.assertFalse(source.cancelled())
        source.cancel()
        self.assertTrue(source.cancelled())
        with self.assertRaises(OperationCancelledError):
            source.token.raise_if_cancelled()
        asyncio.run(source.token.wait())


class RuntimeServerAndAppTests(unittest.TestCase):
    def test_runtime_starts_stops_components_and_exposes_proxy(self):
        component = Component()
        runtime = Runtime(component)

        with runtime:
            self.assertTrue(runtime.started)
            self.assertEqual(runtime.component.component_name, "component")

        self.assertFalse(runtime.started)

    def test_runtime_and_server_reject_empty_or_duplicate_components(self):
        with self.assertRaisesRegex(ValueError, "requires at least"):
            Runtime()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            Runtime(Component(), Component())
        with self.assertRaisesRegex(ValueError, "requires at least"):
            Server()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            Server(Component(), Component())

    def test_server_lazily_builds_app(self):
        server = Server(Component())

        self.assertEqual(server.components[0].component_name, "component")
        self.assertIs(server.app, server.app)

    def test_build_app_health_route_and_lifespan(self):
        component = Component()
        app = build_app([component])

        with TestClient(app) as client:
            self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
