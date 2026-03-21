"""Public exception types for the Trillim SDK."""


class TrillimError(Exception):
    """Base class for public Trillim exceptions."""


class ComponentLifecycleError(TrillimError, RuntimeError):
    """Raised when a component cannot be started or stopped cleanly."""


class OperationCancelledError(TrillimError):
    """Raised when a cancellation token has been triggered."""


class SessionBusyError(TrillimError, RuntimeError):
    """Raised when a single-consumer session is used concurrently."""

