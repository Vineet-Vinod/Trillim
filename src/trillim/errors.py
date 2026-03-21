"""Public exception types for the Trillim SDK."""


class TrillimError(Exception):
    """Base class for public Trillim exceptions."""


class ComponentLifecycleError(TrillimError, RuntimeError):
    """Raised when a component cannot be started or stopped cleanly."""


class OperationCancelledError(TrillimError):
    """Raised when a cancellation token has been triggered."""


class SessionBusyError(TrillimError, RuntimeError):
    """Raised when a single-consumer session is used concurrently."""


class InvalidRequestError(TrillimError, ValueError):
    """Raised when caller input fails validation."""


class ModelValidationError(TrillimError, ValueError):
    """Raised when a model directory is missing required files or metadata."""


class AdmissionRejectedError(TrillimError, RuntimeError):
    """Raised when a component is not admitting more work."""


class ContextOverflowError(TrillimError, ValueError):
    """Raised when a prompt exceeds the active model context window."""

    def __init__(self, token_count: int, limit: int) -> None:
        super().__init__(
            f"Prompt uses {token_count} tokens, which exceeds the context limit of {limit}"
        )
        self.token_count = token_count
        self.limit = limit


class ProgressTimeoutError(TrillimError, TimeoutError):
    """Raised when an operation stops making required progress."""


class SessionClosedError(TrillimError, RuntimeError):
    """Raised when a closed session is used."""


class SessionExhaustedError(TrillimError, RuntimeError):
    """Raised when a session exceeded its lifetime quota."""


class SessionStaleError(TrillimError, RuntimeError):
    """Raised when a session is invalidated by a model swap."""
