"""Tool-call accuracy: expected and forbidden calls, arguments, order and count."""

from __future__ import annotations

from ..matching import call_matches
from ..models import Case, Evaluation, Finding, ToolCall, Trace
from .base import EvalContext, finding, verdict


def _describe(call: ToolCall) -> str:
    return f"{call.tool}({str(call.arguments)[:120]})"


def is_subsequence(wanted: list[str], actual: list[str]) -> bool:
    position = 0
    for name in actual:
        if position < len(wanted) and name == wanted[position]:
            position += 1
    return position == len(wanted)


class ToolCallAccuracy:
    """Compares the recorded tool calls with what the case requires and forbids.

    * A **required** call is satisfied only by a *successful* call with matching arguments.
    * An **attempted** call is satisfied by any call with matching arguments, even one that
      failed. Use it when the right tool is expected to fail (an outage, a missing record).
    * A **forbidden** call is violated by any *attempt*, even one that failed.
    * ``order`` must appear as a subsequence of the successful calls.
    * Calls rejected for invalid arguments or unknown tools are errors in their own right.

    ``precision`` and ``recall`` are reported per case: recall is the share of required calls
    that were made, precision the share of successful calls that matched a required call.
    Repeating a call that failed (a retry) does not lower precision.
    """

    name = "tool_call_accuracy"

    def applies(self, case: Case, trace: Trace) -> bool:
        return case.expected.tool_calls is not None

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        expected = case.expected.tool_calls
        if expected is None:
            return verdict(self.name, [])
        findings: list[Finding] = []
        successful = [call for call in trace.tool_calls if call.ok]

        satisfied = 0
        matched: set[int] = set()
        for spec in expected.required:
            hits = [call for call in successful if call_matches(spec, call)]
            matched.update(call.index for call in hits)
            if len(hits) < spec.min_count:
                same_tool = [call for call in trace.tool_calls if call.tool == spec.tool]
                if not same_tool:
                    message = f"required call to {spec.tool} was never made"
                elif not any(call.ok for call in same_tool):
                    message = f"required call to {spec.tool} never succeeded"
                else:
                    message = (
                        f"{spec.tool} was called, but not with the required arguments "
                        f"{spec.args}; saw {_describe(same_tool[0])}"
                    )
                findings.append(finding("missing_call", message))
            elif spec.max_count is not None and len(hits) > spec.max_count:
                findings.append(
                    finding(
                        "too_many_calls",
                        f"{spec.tool} matched {len(hits)} times, at most {spec.max_count} allowed",
                    )
                )
            else:
                satisfied += 1

        attempted = 0
        for spec in expected.attempted:
            if any(call_matches(spec, call) for call in trace.tool_calls):
                attempted += 1
            else:
                findings.append(
                    finding(
                        "missing_call",
                        f"the agent never tried {spec.tool} with the arguments {spec.args}",
                    )
                )

        for spec in expected.forbidden:
            for call in trace.tool_calls:
                if call_matches(spec, call):
                    findings.append(
                        finding("forbidden_call", f"forbidden call attempted: {_describe(call)}")
                    )
                    break

        for call in trace.tool_calls:
            if call.error is not None and call.error.code == "invalid_arguments":
                findings.append(
                    finding(
                        "invalid_arguments",
                        f"{_describe(call)} was rejected: {call.error.message[:200]}",
                    )
                )
            elif call.error is not None and call.error.code == "unknown_tool":
                findings.append(finding("unknown_tool", f"no such tool: {call.tool}"))

        if expected.order and not is_subsequence(expected.order, [c.tool for c in successful]):
            findings.append(
                finding(
                    "wrong_order",
                    f"expected the order {' -> '.join(expected.order)}, "
                    f"saw {' -> '.join(call.tool for call in successful) or 'no successful calls'}",
                )
            )

        extra = [call for call in successful if call.index not in matched]
        if not expected.allow_extra and extra:
            findings.append(
                finding("unexpected_call", f"call not required by the task: {_describe(extra[0])}")
            )
        if expected.max_calls is not None and len(trace.tool_calls) > expected.max_calls:
            findings.append(
                finding(
                    "too_many_calls",
                    f"{len(trace.tool_calls)} tool calls, at most {expected.max_calls} allowed",
                )
            )

        wanted = len(expected.required) + len(expected.attempted)
        recall = (satisfied + attempted) / wanted if wanted else 1.0
        precision = (len(successful) - len(extra)) / len(successful) if successful else 1.0
        if not expected.required:
            precision = 1.0
        return verdict(
            self.name,
            findings,
            score=recall if findings else 1.0,
            details={
                "required": wanted,
                "required_satisfied": satisfied + attempted,
                "successful_calls": len(successful),
                "matched_calls": len(successful) - len(extra),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
            },
        )
