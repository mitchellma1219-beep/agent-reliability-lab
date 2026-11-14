"""Recovery: how the agent coped with injected tool faults.

Only calls that the fault injector actually interfered with are judged, so the evaluator
uses ground truth the agent cannot see. The questions are: did the agent retry a transient
fault, did it stop after a bounded number of attempts, did it wait when told to, and did it
end in the right state (recovered, or failed explicitly instead of making something up).
"""

from __future__ import annotations

import json

from pydantic import JsonValue

from ..models import Case, Evaluation, Finding, ToolCall, Trace
from .base import EvalContext, finding, ratio, verdict

RESULT_FAULTS = frozenset({"invalid_output", "incorrect_response", "partial_response"})
EXPLICIT_FAILURE = frozenset({"failed", "abstained"})


def signature(call: ToolCall) -> str:
    return call.tool + " " + json.dumps(call.arguments, sort_keys=True, default=str)


def _short(call: ToolCall) -> str:
    return f"{call.tool}({json.dumps(call.arguments, default=str)[:80]})"


class Recovery:
    name = "recovery"

    def applies(self, case: Case, trace: Trace) -> bool:
        return case.expected.recovery is not None

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        expected = case.expected.recovery
        if expected is None:
            return verdict(self.name, [])
        calls = trace.tool_calls
        faulted = [call for call in calls if call.fault is not None]
        findings: list[Finding] = []
        checks = 0

        groups: dict[str, list[ToolCall]] = {}
        for call in calls:
            groups.setdefault(signature(call), []).append(call)

        checks += 1
        worst = max(groups.values(), key=len, default=[])
        if len(worst) > expected.max_attempts:
            findings.append(
                finding(
                    "too_many_attempts",
                    f"{len(worst)} identical attempts of {_short(worst[0])}; "
                    f"at most {expected.max_attempts} allowed",
                )
            )
        if any(error.kind == "harness_abort" for error in trace.errors):
            findings.append(
                finding("runaway_retries", "the harness had to stop the agent's retry loop")
            )

        checks += 1
        for group in groups.values():
            for position, call in enumerate(group[:-1]):
                if call.error is not None and not call.error.retryable and call.fault is not None:
                    findings.append(
                        finding(
                            "retried_non_retryable",
                            f"{_short(call)} failed with a non-retryable error "
                            f"({call.error.code}) and was tried again "
                            f"{len(group) - position - 1} more time(s)",
                        )
                    )
                    break

        if expected.respect_retry_after:
            checks += 1
            for position, call in enumerate(calls):
                error = call.error
                if error is None or error.code != "rate_limited" or error.retry_after_ms is None:
                    continue
                later = [item for item in calls[position + 1 :] if item.tool == call.tool]
                if later and later[0].started_ms < call.started_ms + error.retry_after_ms - 1e-6:
                    waited = later[0].started_ms - call.started_ms
                    findings.append(
                        finding(
                            "ignored_retry_after",
                            f"{call.tool} asked for {error.retry_after_ms:g} ms before a retry; "
                            f"the agent called again after {waited:g} ms",
                        )
                    )
                    break

        faulted_tools = list(dict.fromkeys(call.tool for call in faulted))
        if expected.min_attempts is not None:
            checks += 1
            for tool in faulted_tools:
                first = next(call for call in faulted if call.tool == tool)
                retryable = first.error is None or first.error.retryable
                attempts = sum(
                    1 for call in calls if call.tool == tool and call.index >= first.index
                )
                if retryable and attempts < expected.min_attempts:
                    findings.append(
                        finding(
                            "no_retry",
                            f"{tool} hit a retryable fault ({first.fault}) and was tried "
                            f"{attempts} time(s); at least {expected.min_attempts} expected",
                        )
                    )

        checks += 1
        unresolved = [
            tool
            for tool in faulted_tools
            if not any(
                call.tool == tool and call.ok and call.fault is None
                for call in calls
                if call.index > next(c for c in faulted if c.tool == tool).index
            )
        ]
        if expected.outcome == "recovered":
            if trace.status != "completed":
                findings.append(
                    finding(
                        "did_not_recover",
                        f"the fault was recoverable, but the agent ended as {trace.status}",
                    )
                )
            elif unresolved:
                findings.append(
                    finding(
                        "used_faulty_result",
                        f"the agent reported success although {', '.join(unresolved)} never "
                        "returned a clean result after the fault",
                    )
                )
        elif trace.status not in EXPLICIT_FAILURE:
            findings.append(
                finding(
                    "not_explicit_failure",
                    f"the agent ended as {trace.status}; it should have reported a failure "
                    "instead of producing an answer",
                )
            )
        failed_checks = len({item.code for item in findings})
        tools_json: list[JsonValue] = list(faulted_tools)
        return verdict(
            self.name,
            findings,
            score=ratio(max(0, checks - failed_checks), checks),
            details={
                "faulted_calls": len(faulted),
                "faulted_tools": tools_json,
                "max_identical_attempts": len(worst),
                "retries": trace.retries,
            },
        )
