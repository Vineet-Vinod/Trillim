"""Tests for fixed STT configuration."""

from pathlib import Path
import unittest

from trillim.components.stt._config import (
    DEFAULT_WORKER_CONFIG,
    OwnedAudioInput,
    SourceFileSnapshot,
)


class STTConfigTests(unittest.TestCase):
    def test_owned_audio_input_records_path_and_size(self):
        owned = OwnedAudioInput(path=Path("/tmp/audio"), size_bytes=12)
        self.assertEqual(owned.path, Path("/tmp/audio"))
        self.assertEqual(owned.size_bytes, 12)

    def test_source_file_snapshot_is_value_comparable(self):
        self.assertEqual(
            SourceFileSnapshot(size_bytes=1, modified_ns=2),
            SourceFileSnapshot(size_bytes=1, modified_ns=2),
        )

    def test_worker_config_is_fixed_and_cpu_bound(self):
        self.assertEqual(DEFAULT_WORKER_CONFIG.model_name, "base")
        self.assertEqual(DEFAULT_WORKER_CONFIG.device, "cpu")
        self.assertEqual(DEFAULT_WORKER_CONFIG.compute_type, "int8")
