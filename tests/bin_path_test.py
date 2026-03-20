# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for bundled binary path resolution."""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trillim._bin_path as bin_path


class BinaryPathResolutionTests(unittest.TestCase):
    def test_resolve_prefers_env_override_when_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / "trillim-quantize"
            binary.write_text("binary", encoding="utf-8")
            binary.chmod(0o644)

            with (
                patch.dict(os.environ, {"TRILLIM_QUANTIZE_BIN": str(binary)}, clear=False),
                patch.object(bin_path, "_BIN_DIR", str(Path(temp_dir) / "empty-bin-dir")),
                patch.object(bin_path, "_EXE_SUFFIX", ""),
            ):
                resolved = bin_path._resolve("trillim-quantize")

            self.assertEqual(resolved, str(binary))
            self.assertTrue(os.access(binary, os.X_OK))

    def test_resolve_raises_for_missing_env_override_target(self):
        with patch.dict(os.environ, {"TRILLIM_INFERENCE_BIN": "/missing/bin"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "TRILLIM_INFERENCE_BIN is set"):
                bin_path._resolve("inference")

    def test_resolve_returns_primary_binary_and_marks_it_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / "inference"
            binary.write_text("binary", encoding="utf-8")
            binary.chmod(0o644)

            with (
                patch.object(bin_path, "_BIN_DIR", temp_dir),
                patch.object(bin_path, "_EXE_SUFFIX", ""),
            ):
                resolved = bin_path._resolve("inference")

            self.assertEqual(resolved, str(binary))
            self.assertTrue(os.access(binary, os.X_OK))

    def test_resolve_falls_back_to_unsuffixed_binary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / "trillim-quantize"
            binary.write_text("binary", encoding="utf-8")
            binary.chmod(0o644)

            with (
                patch.object(bin_path, "_BIN_DIR", temp_dir),
                patch.object(bin_path, "_EXE_SUFFIX", ".exe"),
            ):
                resolved = bin_path._resolve("trillim-quantize")

            self.assertEqual(resolved, str(binary))
            self.assertTrue(binary.stat().st_mode & stat.S_IXUSR)

    def test_resolve_raises_source_build_error_when_bin_dir_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(bin_path, "_BIN_DIR", temp_dir),
                patch.object(bin_path, "_EXE_SUFFIX", ""),
                patch.dict(bin_path._SOURCE_BINARIES, {"inference": ("TRILLIM_INFERENCE_BIN", Path(temp_dir) / "missing-source-bin")}, clear=False),
            ):
                with self.assertRaisesRegex(RuntimeError, "No packaged binaries found"):
                    bin_path._resolve("inference")

    def test_resolve_falls_back_to_source_tree_binary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / "trillim-inference"
            binary.write_text("binary", encoding="utf-8")
            binary.chmod(0o644)

            with (
                patch.object(bin_path, "_BIN_DIR", str(Path(temp_dir) / "empty-bin-dir")),
                patch.object(bin_path, "_EXE_SUFFIX", ""),
                patch.dict(bin_path._SOURCE_BINARIES, {"inference": ("TRILLIM_INFERENCE_BIN", binary)}, clear=False),
            ):
                resolved = bin_path._resolve("inference")

            self.assertEqual(resolved, str(binary))
            self.assertTrue(os.access(binary, os.X_OK))

    def test_resolve_raises_platform_error_when_requested_binary_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "other-binary").write_text("binary", encoding="utf-8")

            with (
                patch.object(bin_path, "_BIN_DIR", temp_dir),
                patch.object(bin_path, "_EXE_SUFFIX", ""),
            ):
                with self.assertRaisesRegex(RuntimeError, "may not be supported"):
                    bin_path._resolve("missing")

    def test_public_helpers_delegate_to_resolve(self):
        with patch("trillim._bin_path._resolve", side_effect=["/a", "/b"]) as mock_resolve:
            self.assertEqual(bin_path.inference_bin(), "/a")
            self.assertEqual(bin_path.quantize_bin(), "/b")

        self.assertEqual(mock_resolve.call_args_list[0].args, ("inference",))
        self.assertEqual(mock_resolve.call_args_list[1].args, ("trillim-quantize",))


if __name__ == "__main__":
    unittest.main()
