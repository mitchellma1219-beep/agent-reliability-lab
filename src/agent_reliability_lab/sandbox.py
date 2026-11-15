"""The mediation layer between an agent and the simulated tools.

The sandbox is the only way an agent can reach a tool. It validates arguments against the
tool's JSON Schema, applies injected faults, keeps simulated time, enforces budgets and
records every call. The trace is written here, by the harness, so an agent cannot
misreport what it did.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import HarnessAbort, ToolError
from .faults import FaultInjector, apply_mutation
from .models import (
    LlmCall,
    ModelUsage,
    SecurityEvent,
    ToolCall,
    ToolErrorInfo,
    Usage,
    Wait,
)
from .schema import cached_validator, validate
from .tools import CallContext, ToolInfo, ToolSpec, World

MAX_WAIT_MS = 3_600_000.0


@dataclass(frozen=True)
class Limits:
    max_tool_calls: int = 50
    max_llm_calls: int = 200
    max_argument_bytes: int = 65_536
    max_result_chars: int = 20_000


class Sandbox:
    """One case run's view of the simulated tools, with recording and limits."""

    def __init__(
        self,
        world: World,
        tools: Mapping[str, ToolSpec],
        injector: FaultInjector | None = None,
        limits: Limits | None = None,
    ) -> None:
        self._world = world
        self._tools = dict(tools)
        self._injector = injector or FaultInjector([], 0)
        self._limits = limits or Limits()
        self._lock = threading.RLock()
        self._index = 0
        self._attempts: dict[str, int] = {}
        self._needs_retry: dict[str, bool] = {}
        self._closed: str | None = None
        self.now_ms = 0.0
        self.abort_reason: str | None = None
        self.tool_calls: list[ToolCall] = []
        self.llm_calls: list[LlmCall] = []
        self.waits: list[Wait] = []
        self.events: list[SecurityEvent] = []
        self.annotations: dict[str, Any] = {}

    # ------------------------------------------------------------------ lifecycle

    @property
    def world(self) -> World:
        return self._world

    def close(self, reason: str) -> None:
        """Refuse all further activity, for example after a timeout."""
        with self._lock:
            if self._closed is None:
                self._closed = reason

    def _check_open(self) -> None:
        if self._closed is not None:
            raise HarnessAbort("closed", f"the case has ended ({self._closed})")

    def _next_index(self) -> int:
        self._index += 1
        return self._index - 1

    # ------------------------------------------------------------------ tool access

    def infos(self) -> list[ToolInfo]:
        return [spec.info() for spec in self._tools.values()]

    def spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Call a simulated tool. Raises :class:`ToolError` to the agent on failure."""
        with self._lock:
            return self._call(name, arguments)

    def _call(self, name: str, arguments: Mapping[str, Any] | None) -> Any:
        self._check_open()
        if len(self.tool_calls) >= self._limits.max_tool_calls:
            self.abort_reason = "tool_call_budget"
            raise HarnessAbort(
                "tool_call_budget",
                f"the agent made more than {self._limits.max_tool_calls} tool calls",
            )
        tool_name = name if isinstance(name, str) else repr(name)
        tool_name = tool_name[:100]
        recorded, problem = self._normalise(arguments)
        spec = self._tools.get(tool_name)
        if spec is None:
            error = ToolError("unknown_tool", f"no tool named {tool_name!r} is available")
            self._record(tool_name, recorded, error=error)
            raise error
        if problem is None:
            violations = validate(
                spec.parameters, recorded, validator=cached_validator(spec.parameters)
            )
            if violations:
                problem = "; ".join(violations[:3])
        if problem is not None:
            error = ToolError("invalid_arguments", problem)
            self._record(tool_name, recorded, spec=spec, error=error)
            raise error

        decision = self._injector.decide(
            tool_name,
            depends_on=spec.depends_on,
            supports_partial=spec.supports_partial,
            now_ms=self.now_ms,
        )
        if decision.error is not None:
            latency = spec.latency_ms if decision.delay_ms is None else decision.delay_ms
            self._record(
                tool_name,
                recorded,
                spec=spec,
                error=decision.error,
                latency_ms=latency,
                fault=decision.kind,
            )
            raise decision.error

        latency = spec.latency_ms + (decision.delay_ms or 0.0)
        context = CallContext(world=self._world, partial=decision.partial, events=self.events)
        try:
            result = spec.handler(context, json.loads(json.dumps(recorded)))
            if decision.mutation is not None:
                result = apply_mutation(result, decision.mutation)
            result = json.loads(json.dumps(result))
        except ToolError as exc:
            self._record(
                tool_name, recorded, spec=spec, error=exc, latency_ms=latency, fault=decision.kind
            )
            raise
        except Exception as exc:
            error = ToolError("internal_error", f"the tool failed: {type(exc).__name__}")
            self._record(
                tool_name, recorded, spec=spec, error=error, latency_ms=latency, fault=decision.kind
            )
            raise error from exc
        self._record(
            tool_name, recorded, spec=spec, result=result, latency_ms=latency, fault=decision.kind
        )
        return result

    def _normalise(self, arguments: Mapping[str, Any] | None) -> tuple[Any, str | None]:
        """Return a JSON copy of the arguments, or a printable stand-in plus a problem."""
        if arguments is None:
            return {}, None
        if not isinstance(arguments, Mapping):
            return repr(arguments)[:500], "arguments must be an object"
        try:
            text = json.dumps(dict(arguments), allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            return repr(arguments)[:500], "arguments are not valid JSON"
        if len(text.encode("utf-8")) > self._limits.max_argument_bytes:
            return (
                {"_truncated": True, "preview": text[:500]},
                f"arguments are larger than {self._limits.max_argument_bytes} bytes",
            )
        return json.loads(text), None

    def _record(
        self,
        tool: str,
        arguments: Any,
        *,
        spec: ToolSpec | None = None,
        result: Any = None,
        error: ToolError | None = None,
        latency_ms: float = 0.0,
        fault: str | None = None,
    ) -> None:
        signature = tool + "\x00" + json.dumps(arguments, sort_keys=True, default=repr)
        attempt = self._attempts.get(signature, 0) + 1
        self._attempts[signature] = attempt
        is_retry = attempt > 1 and self._needs_retry.get(signature, False)
        self._needs_retry[signature] = error is not None or fault is not None

        truncated = False
        stored = result
        if error is None:
            text = json.dumps(result)
            if len(text) > self._limits.max_result_chars:
                stored = {"_truncated": True, "preview": text[:2000]}
                truncated = True
        info = None
        if error is not None:
            info = ToolErrorInfo(
                code=error.code,
                message=error.message[:500],
                retryable=error.retryable,
                retry_after_ms=error.retry_after_ms,
            )
        self.tool_calls.append(
            ToolCall(
                index=self._next_index(),
                tool=tool,
                arguments=arguments,
                ok=error is None,
                result=stored,
                result_truncated=truncated,
                error=info,
                attempt=attempt,
                is_retry=is_retry,
                latency_ms=latency_ms,
                started_ms=self.now_ms,
                fault=fault,
                sink=bool(spec and spec.sink),
                side_effect=bool(spec and spec.side_effect),
            )
        )
        self.now_ms += latency_ms

    # ------------------------------------------------------------------ usage and time

    def record_llm_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        note: str = "",
    ) -> None:
        with self._lock:
            self._check_open()
            if len(self.llm_calls) >= self._limits.max_llm_calls:
                self.abort_reason = "step_budget"
                raise HarnessAbort(
                    "step_budget",
                    f"the agent reported more than {self._limits.max_llm_calls} model calls",
                )
            if input_tokens < 0 or output_tokens < 0 or latency_ms < 0:
                raise ValueError("token counts and latency must not be negative")
            self.llm_calls.append(
                LlmCall(
                    index=self._next_index(),
                    model=str(model)[:100],
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=float(latency_ms),
                    started_ms=self.now_ms,
                    note=str(note)[:200],
                )
            )
            self.now_ms += float(latency_ms)

    def wait(self, ms: float, reason: str = "") -> None:
        """Advance simulated time, for example to back off before a retry."""
        with self._lock:
            self._check_open()
            if not 0 <= ms <= MAX_WAIT_MS:
                raise ValueError("wait must be between 0 and one hour of simulated time")
            self.waits.append(
                Wait(
                    index=self._next_index(),
                    ms=float(ms),
                    started_ms=self.now_ms,
                    reason=reason[:200],
                )
            )
            self.now_ms += float(ms)

    def annotate(self, key: str, value: Any) -> None:
        """Attach a small piece of agent-supplied metadata to the trace."""
        with self._lock:
            self._check_open()
            if len(self.annotations) < 50 or key in self.annotations:
                self.annotations[str(key)[:100]] = json.loads(json.dumps(value, default=str))

    def usage(self) -> Usage:
        by_model: dict[str, ModelUsage] = {}
        for call in self.llm_calls:
            entry = by_model.setdefault(call.model, ModelUsage())
            entry.input_tokens += call.input_tokens
            entry.output_tokens += call.output_tokens
            entry.calls += 1
        return Usage(
            input_tokens=sum(item.input_tokens for item in by_model.values()),
            output_tokens=sum(item.output_tokens for item in by_model.values()),
            llm_calls=len(self.llm_calls),
            by_model=by_model,
        )
