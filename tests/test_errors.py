"""Tests for public error types."""

import unittest

from trillim.errors import (
    ComponentLifecycleError,
    OperationCancelledError,
    SessionBusyError,
    TrillimError,
)


class ErrorTests(unittest.TestCase):
    def test_public_errors_are_trillim_errors(self):
        self.assertTrue(issubclass(ComponentLifecycleError, TrillimError))
        self.assertTrue(issubclass(OperationCancelledError, TrillimError))
        self.assertTrue(issubclass(SessionBusyError, TrillimError))

