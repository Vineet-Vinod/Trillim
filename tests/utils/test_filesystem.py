"""Tests for filesystem helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trillim.utils.filesystem import (
    atomic_write_bytes,
    canonicalize_path,
    ensure_within_root,
    unlink_if_exists,
)


class FilesystemTests(unittest.TestCase):
    def test_canonicalize_and_ensure_within_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child = root / "child" / "file.txt"
            child.parent.mkdir()
            child.write_text("x", encoding="utf-8")
            self.assertEqual(canonicalize_path(child, strict=True), child.resolve())
            self.assertEqual(ensure_within_root(child, root, strict=True), child.resolve())
            with self.assertRaisesRegex(ValueError, "outside allowed root"):
                ensure_within_root(Path(temp_dir).parent, root, strict=False)

    def test_atomic_write_bytes_writes_and_cleans_up_on_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "data.bin"
            atomic_write_bytes(target, b"payload")
            self.assertEqual(target.read_bytes(), b"payload")

            broken_target = Path(temp_dir) / "broken.bin"
            with patch("trillim.utils.filesystem.os.replace", side_effect=OSError("nope")):
                with self.assertRaises(OSError):
                    atomic_write_bytes(broken_target, b"payload")
            leftovers = [item for item in Path(temp_dir).iterdir() if item.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_unlink_if_exists_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "demo.txt"
            path.write_text("demo", encoding="utf-8")
            unlink_if_exists(path)
            unlink_if_exists(path)
            self.assertFalse(path.exists())

