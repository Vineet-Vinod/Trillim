"""Tests for fixed STT internal limits."""

import unittest

from trillim.components.stt import _limits


class STTLimitsTests(unittest.TestCase):
    def test_limits_are_positive_and_internally_consistent(self):
        self.assertGreater(_limits.MAX_UPLOAD_BYTES, 0)
        self.assertGreater(_limits.SPOOL_CHUNK_SIZE_BYTES, 0)
        self.assertLessEqual(_limits.SPOOL_CHUNK_SIZE_BYTES, _limits.MAX_UPLOAD_BYTES)
        self.assertGreater(_limits.MAX_LANGUAGE_CHARS, 0)
        self.assertGreater(_limits.UPLOAD_PROGRESS_TIMEOUT_SECONDS, 0)
        self.assertGreaterEqual(
            _limits.TOTAL_UPLOAD_TIMEOUT_SECONDS,
            _limits.UPLOAD_PROGRESS_TIMEOUT_SECONDS,
        )

    def test_admission_and_timeout_limits_match_phase_4_contract(self):
        self.assertEqual(_limits.MAX_ACTIVE_TRANSCRIPTIONS, 1)
        self.assertEqual(_limits.MAX_QUEUED_TRANSCRIPTIONS, 0)
        self.assertEqual(_limits.TOTAL_TRANSCRIPTION_TIMEOUT_SECONDS, 180)
