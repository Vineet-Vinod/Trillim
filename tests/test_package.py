"""Tests for top-level package exports."""

import unittest

import trillim
from trillim import (
    AdmissionRejectedError,
    ContextOverflowError,
    InvalidRequestError,
    LLM,
    ModelValidationError,
    ProgressTimeoutError,
    Runtime,
    STT,
    Server,
    SessionClosedError,
    SessionExhaustedError,
    SessionStaleError,
    TTS,
)


class PackageExportTests(unittest.TestCase):
    def test_top_level_exports_exist(self):
        self.assertIs(trillim.LLM, LLM)
        self.assertIs(trillim.STT, STT)
        self.assertIs(trillim.TTS, TTS)
        self.assertIs(trillim.Runtime, Runtime)
        self.assertIs(trillim.Server, Server)
        self.assertIs(trillim.InvalidRequestError, InvalidRequestError)
        self.assertIs(trillim.ModelValidationError, ModelValidationError)
        self.assertIs(trillim.AdmissionRejectedError, AdmissionRejectedError)
        self.assertIs(trillim.ContextOverflowError, ContextOverflowError)
        self.assertIs(trillim.ProgressTimeoutError, ProgressTimeoutError)
        self.assertIs(trillim.SessionClosedError, SessionClosedError)
        self.assertIs(trillim.SessionExhaustedError, SessionExhaustedError)
        self.assertIs(trillim.SessionStaleError, SessionStaleError)
