"""Seeded, deterministic fault injection for simulated tools.

Each fault kind wraps a tool call and either raises a :class:`~.errors.ToolError` before the
handler runs or alters the handler's result afterwards:

==================== ================================================================
``timeout``          the call takes ``delay_ms`` of simulated time, then fails (retryable)
``rate_limit``       fails with ``retry_after_ms``; calling again too early fails again
``transient_error``  fails with a retryable error for the first ``times`` calls
``permanent_error``  fails with a non-retryable error
``retry_exhaustion`` fails with a retryable error on every call
``dependency_failure`` every tool that depends on the named service fails (non-retryable)
``incorrect_response`` succeeds, but one field of the result is silently wrong
``partial_response`` a batch tool completes only part of the work and says so
``invalid_output``   the result is malformed (truncated JSON text, wrong type, null)
==================== ================================================================

Probabilistic faults do not use a random number generator with hidden state. Whether call
``n`` to tool ``t`` is affected is a pure function of ``(seed, fault index, tool, n)``, so
the same seed gives the same faults no matter how the agent orders its other calls.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .errors import ToolError
from .models import FaultSpec, Mutation

DEFAULT_TIMEOUT_MS = 10_000.0


def derive_seed(*parts: object) -> int:
    """A 63-bit seed derived from the parts; stable across processes and platforms."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def uniform(*parts: object) -> float:
    """A number in [0, 1) that depends only on the parts (a counter-based generator)."""
    return derive_seed(*parts) / float(1 << 63)


@dataclass
class FaultDecision:
    """What the injector decided for one call."""

    kind: str | None = None
    error: ToolError | None = None
    delay_ms: float | None = None
    partial: bool = False
    mutation: Mutation | None = None


def _get_path(value: Any, path: list[str]) -> tuple[Any, str] | None:
    """Return the container holding the last path segment, or ``None`` if the path is absent."""
    current = value
    for segment in path[:-1]:
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return None
    if isinstance(current, dict) and path[-1] in current:
        return current, path[-1]
    return None


def apply_mutation(result: Any, mutation: Mutation) -> Any:
    """Return a corrupted copy of ``result``. The original is not modified."""
    value = json.loads(json.dumps(result))
    if mutation.op == "null":
        return None
    if mutation.op == "stringify":
        return json.dumps(value)
    if mutation.op == "truncate_json":
        text = json.dumps(value)
        return text[: max(1, int(len(text) * 0.6))]
    if mutation.op == "wrong_type":
        return (
            [[key, item] for key, item in value.items()] if isinstance(value, dict) else str(value)
        )
    located = _get_path(value, (mutation.field or "").split("."))
    if located is None:
        return value
    container, key = located
    if mutation.op == "drop_field":
        del container[key]
    elif mutation.op == "set":
        container[key] = mutation.value
    else:
        current = container[key]
        if isinstance(current, int | float) and not isinstance(current, bool):
            factor = -1.0 if mutation.op == "negate" else float(mutation.factor or 1.0)
            scaled = current * factor
            container[key] = (
                int(scaled) if isinstance(current, int) and scaled == int(scaled) else scaled
            )
    return value


class FaultInjector:
    """Decides, call by call, whether a fault applies."""

    def __init__(self, specs: list[FaultSpec], seed: int) -> None:
        self._specs = list(specs)
        self._seed = seed
        self._seen: dict[tuple[int, str], int] = {}
        self._fired: dict[tuple[int, str], int] = {}
        self._limited_until: dict[str, float] = {}

    def _matches(
        self, spec: FaultSpec, tool: str, depends_on: tuple[str, ...], partial: bool
    ) -> bool:
        if spec.tool not in ("*", tool):
            return False
        if spec.kind == "dependency_failure":
            return spec.dependency in depends_on
        if spec.kind == "partial_response":
            return partial
        return True

    def decide(
        self,
        tool: str,
        *,
        depends_on: tuple[str, ...] = (),
        supports_partial: bool = False,
        now_ms: float = 0.0,
    ) -> FaultDecision:
        until = self._limited_until.get(tool)
        if until is not None and now_ms < until:
            return FaultDecision(
                kind="rate_limit",
                error=ToolError(
                    "rate_limited",
                    "called again before the retry-after period ended",
                    retryable=True,
                    retry_after_ms=until - now_ms,
                ),
            )
        for index, spec in enumerate(self._specs):
            if not self._matches(spec, tool, depends_on, supports_partial):
                continue
            key = (index, tool)
            seen = self._seen.get(key, 0) + 1
            self._seen[key] = seen
            if seen <= spec.after:
                continue
            if spec.times is not None and self._fired.get(key, 0) >= spec.times:
                continue
            if (
                spec.probability < 1.0
                and uniform(self._seed, index, tool, seen) >= spec.probability
            ):
                continue
            self._fired[key] = self._fired.get(key, 0) + 1
            return self._build(spec, tool, now_ms)
        return FaultDecision()

    def _build(self, spec: FaultSpec, tool: str, now_ms: float) -> FaultDecision:
        kind = spec.kind
        if kind == "timeout":
            delay = DEFAULT_TIMEOUT_MS if spec.delay_ms is None else spec.delay_ms
            error = ToolError("timeout", f"{tool} did not answer in time", retryable=True)
            return FaultDecision(kind=kind, error=error, delay_ms=delay)
        if kind == "rate_limit":
            self._limited_until[tool] = now_ms + spec.retry_after_ms
            error = ToolError(
                "rate_limited",
                f"too many requests to {tool}",
                retryable=True,
                retry_after_ms=spec.retry_after_ms,
            )
            return FaultDecision(kind=kind, error=error, delay_ms=spec.delay_ms)
        if kind in {"transient_error", "retry_exhaustion"}:
            error = ToolError("unavailable", f"{tool} is temporarily unavailable", retryable=True)
            return FaultDecision(kind=kind, error=error, delay_ms=spec.delay_ms)
        if kind == "permanent_error":
            error = ToolError(
                "internal_error", f"{tool} failed and will not recover", retryable=False
            )
            return FaultDecision(kind=kind, error=error, delay_ms=spec.delay_ms)
        if kind == "dependency_failure":
            error = ToolError(
                "dependency_failed",
                f"{tool} cannot run because its dependency '{spec.dependency}' is down",
                retryable=False,
            )
            return FaultDecision(kind=kind, error=error, delay_ms=spec.delay_ms)
        if kind == "partial_response":
            return FaultDecision(kind=kind, partial=True, delay_ms=spec.delay_ms)
        mutation = spec.mutation or Mutation(op="truncate_json")
        return FaultDecision(kind=kind, mutation=mutation, delay_ms=spec.delay_ms)


FAULT_PROFILES: dict[str, list[FaultSpec]] = {
    "flaky-tools": [FaultSpec(kind="transient_error", probability=0.15)],
    "rate-limited": [FaultSpec(kind="rate_limit", probability=0.2, retry_after_ms=500.0)],
    "slow-network": [FaultSpec(kind="timeout", probability=0.1, delay_ms=3000.0)],
    "degraded": [
        FaultSpec(kind="transient_error", probability=0.1),
        FaultSpec(kind="rate_limit", probability=0.1, retry_after_ms=500.0),
        FaultSpec(kind="timeout", probability=0.05, delay_ms=3000.0),
    ],
}
