"""Evaluator protocol and helpers.

An evaluator is a deterministic function of a case and a trace. It never calls a model, never
reads the ``sim`` block and never looks at how the agent is implemented.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import JsonValue

from ..models import Case, Document, Evaluation, Finding, PriceTable, Trace
from ..tools import BUILTIN_TOOLS, ToolSpec


@dataclass(frozen=True)
class EvalContext:
    """Shared inputs that are not part of the case or the trace."""

    prices: PriceTable = field(default_factory=PriceTable)
    tools: Mapping[str, ToolSpec] = field(default_factory=lambda: BUILTIN_TOOLS)
    context: tuple[Document, ...] = ()
    estimated_cost_usd: float | None = None


class Evaluator(Protocol):
    name: str

    def applies(self, case: Case, trace: Trace) -> bool:
        """Whether this evaluator has anything to check for the case."""
        ...

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation: ...


def finding(
    code: str, message: str, severity: Literal["error", "warning", "info"] = "error"
) -> Finding:
    return Finding(code=code, message=message[:1000], severity=severity)


def verdict(
    name: str,
    findings: list[Finding],
    *,
    score: float | None = None,
    details: dict[str, JsonValue] | None = None,
) -> Evaluation:
    """Build an evaluation: it passes when there is no finding of severity ``error``."""
    failed = any(item.severity == "error" for item in findings)
    if score is None:
        score = 0.0 if failed else 1.0
    return Evaluation(
        evaluator=name,
        passed=not failed,
        score=round(max(0.0, min(1.0, score)), 6),
        findings=findings,
        details=details or {},
    )


def ratio(passed: int, total: int) -> float:
    return 1.0 if total == 0 else passed / total
