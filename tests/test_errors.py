"""Tests for public error types."""

import unittest

from trillim.errors import (
    AdmissionRejectedError,
    ComponentLifecycleError,
    ContextOverflowError,
    InvalidRequestError,
    ModelValidationError,
    OperationCancelledError,
    ProgressTimeoutError,
    SessionClosedError,
    SessionBusyError,
    SessionExhaustedError,
    SessionStaleError,
    TrillimError,
)


class ErrorTests(unittest.TestCase):
    def test_public_errors_are_trillim_errors(self):
        self.assertTrue(issubclass(AdmissionRejectedError, TrillimError))
        self.assertTrue(issubclass(ComponentLifecycleError, TrillimError))
        self.assertTrue(issubclass(ContextOverflowError, TrillimError))
        self.assertTrue(issubclass(InvalidRequestError, TrillimError))
        self.assertTrue(issubclass(ModelValidationError, TrillimError))
        self.assertTrue(issubclass(OperationCancelledError, TrillimError))
        self.assertTrue(issubclass(ProgressTimeoutError, TrillimError))
        self.assertTrue(issubclass(SessionClosedError, TrillimError))
        self.assertTrue(issubclass(SessionBusyError, TrillimError))
        self.assertTrue(issubclass(SessionExhaustedError, TrillimError))
        self.assertTrue(issubclass(SessionStaleError, TrillimError))

    def test_context_overflow_error_exposes_limits(self):
        error = ContextOverflowError(33, 32)

        self.assertEqual(error.token_count, 33)
        self.assertEqual(error.limit, 32)
        self.assertIn("33", str(error))
