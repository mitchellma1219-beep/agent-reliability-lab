"""Latency and cost budgets.

Latency is the *accounted* time of the run: simulated tool latency, model latency reported by
the agent, and waits. Cost is an **estimate** computed from reported token counts and the
price table in the run configuration; it is only as good as those two inputs.
"""

from __future__ import annotations

from ..models import Case, Evaluation, Finding, Trace
from .base import EvalContext, finding, verdict


class LatencyBudget:
    name = "latency_budget"

    def applies(self, case: Case, trace: Trace) -> bool:
        budget = case.expected.budget
        return budget is not None and budget.max_latency_ms is not None

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        budget = case.expected.budget
        limit = budget.max_latency_ms if budget is not None else None
        findings: list[Finding] = []
        if limit is not None and trace.latency_ms > limit:
            findings.append(
                finding(
                    "latency_exceeded",
                    f"accounted latency {trace.latency_ms:,.0f} ms exceeds the budget of "
                    f"{limit:,.0f} ms",
                )
            )
        score = 1.0 if limit is None else min(1.0, limit / max(trace.latency_ms, 1e-9))
        return verdict(
            self.name,
            findings,
            score=score,
            details={"latency_ms": round(trace.latency_ms, 3), "budget_ms": limit},
        )


class CostBudget:
    name = "cost_budget"

    def applies(self, case: Case, trace: Trace) -> bool:
        budget = case.expected.budget
        return budget is not None and (
            budget.max_cost_usd is not None
            or budget.max_tokens is not None
            or budget.max_tool_calls is not None
        )

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        budget = case.expected.budget
        findings: list[Finding] = []
        cost = ctx.estimated_cost_usd
        tokens = trace.usage.input_tokens + trace.usage.output_tokens
        if budget is not None and budget.max_cost_usd is not None:
            if cost is None:
                models = ", ".join(sorted(trace.usage.by_model)) or "none"
                findings.append(
                    finding(
                        "cost_unknown",
                        f"no price for the model(s) used ({models}); the cost budget cannot "
                        "be checked",
                    )
                )
            elif cost > budget.max_cost_usd:
                findings.append(
                    finding(
                        "cost_exceeded",
                        f"estimated cost ${cost:.4f} exceeds the budget of "
                        f"${budget.max_cost_usd:.4f}",
                    )
                )
        if budget is not None and budget.max_tokens is not None and tokens > budget.max_tokens:
            findings.append(
                finding(
                    "tokens_exceeded", f"{tokens:,} tokens used, budget is {budget.max_tokens:,}"
                )
            )
        if (
            budget is not None
            and budget.max_tool_calls is not None
            and len(trace.tool_calls) > budget.max_tool_calls
        ):
            findings.append(
                finding(
                    "tool_calls_exceeded",
                    f"{len(trace.tool_calls)} tool calls, budget is {budget.max_tool_calls}",
                )
            )
        return verdict(
            self.name,
            findings,
            details={
                "estimated_cost_usd": None if cost is None else round(cost, 8),
                "tokens": tokens,
                "tool_calls": len(trace.tool_calls),
            },
        )
