"""Exception types.

Three families with different audiences:

* :class:`ArlError` and its subclasses are raised to the *person running the lab* (bad suite
  file, unsafe run name, missing opt-in). The CLI prints them without a traceback.
* :class:`ToolError` is raised to the *agent under test* by a simulated tool. Handling it well
  is part of what the lab measures.
* :class:`HarnessAbort` stops a runaway agent. It derives from ``BaseException`` on purpose, so
  that an agent's ``except Exception`` retry loop cannot swallow it.
"""

from __future__ import annotations


class ArlError(Exception):
    """Base class for errors that are reported to the user without a traceback."""


class SuiteError(ArlError):
    """A suite file is missing, malformed, too large or semantically invalid."""


class StorageError(ArlError):
    """A run name or path is unsafe, or a stored run cannot be read."""


class ConfigError(ArlError):
    """A run, price table, agent reference or judge is configured incorrectly."""


class JudgeError(ArlError):
    """A judge backend could not be used or returned something unusable."""


class ToolError(Exception):
    """An error returned by a simulated tool to the agent.

    ``retryable`` tells the agent whether trying again can help; ``retry_after_ms`` is set for
    rate limits. ``code`` is a short stable identifier such as ``timeout`` or ``not_found``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        retry_after_ms: float | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms


class HarnessAbort(BaseException):
    """Raised inside the agent's thread when the harness stops the case.

    Reasons: ``tool_call_budget`` (too many tool calls), ``step_budget`` (too many recorded
    model steps) and ``closed`` (the case already timed out or finished).
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
