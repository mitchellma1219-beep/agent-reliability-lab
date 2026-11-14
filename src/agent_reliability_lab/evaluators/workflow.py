"""Multi-step workflow: every step done, in dependency order, leaving the right state."""

from __future__ import annotations

from typing import Any

from ..matching import args_match, values_equal
from ..models import Case, Evaluation, Finding, StateAssertion, Trace
from .base import EvalContext, finding, ratio, verdict

_MISSING = object()


def lookup_state(state: Any, path: list[str | int]) -> Any:
    """Follow ``path`` through dicts and lists; returns a private marker when absent."""
    current = state
    for part in path:
        if isinstance(current, dict) and str(part) in current:
            current = current[str(part)]
        elif (
            isinstance(current, list)
            and isinstance(part, int)
            and -len(current) <= part < len(current)
        ):
            current = current[part]
        else:
            return _MISSING
    return current


def check_assertion(assertion: StateAssertion, state: Any) -> str | None:
    """Return a description of the mismatch, or ``None`` if the assertion holds."""
    where = "/".join(str(part) for part in assertion.path)
    value = lookup_state(state, assertion.path)
    if assertion.exists is not None:
        present = value is not _MISSING
        return (
            None if present == assertion.exists else f"{where}: expected exists={assertion.exists}"
        )
    if value is _MISSING:
        return f"{where}: no such entry in the final state"
    if assertion.length is not None:
        size = len(value) if isinstance(value, list | dict | str) else None
        return (
            None
            if size == assertion.length
            else f"{where}: length is {size}, expected {assertion.length}"
        )
    if assertion.contains is not None:
        text = value if isinstance(value, str) else str(value)
        return (
            None
            if assertion.contains.casefold() in text.casefold()
            else (f"{where}: does not contain {assertion.contains!r}")
        )
    if not values_equal(assertion.equals, value):
        return f"{where}: is {str(value)[:80]!r}, expected {assertion.equals!r}"
    return None


class Workflow:
    """Checks the steps of a workflow against the successful tool calls and the final state.

    A step is done when a *successful* call to its tool with matching arguments exists. A
    step listed in ``after`` must have been done earlier than the step that depends on it.
    ``final_state`` assertions are evaluated on the simulated world as it was when the run
    ended, which catches work that was reported but never carried out.
    """

    name = "workflow"

    def applies(self, case: Case, trace: Trace) -> bool:
        return case.expected.workflow is not None

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        expected = case.expected.workflow
        if expected is None:
            return verdict(self.name, [])
        findings: list[Finding] = []
        first_done: dict[str, int] = {}
        for step in expected.steps:
            for call in trace.tool_calls:
                if call.ok and call.tool == step.tool and args_match(step.args, call.arguments):
                    first_done[step.id] = call.index
                    break
        completed = 0
        for step in expected.steps:
            if step.id not in first_done:
                findings.append(
                    finding("missing_step", f"step {step.id!r} ({step.tool}) was not carried out")
                )
                continue
            late = [
                dependency
                for dependency in step.after
                if dependency in first_done and first_done[dependency] > first_done[step.id]
            ]
            if late:
                findings.append(
                    finding(
                        "step_out_of_order",
                        f"step {step.id!r} ran before the step(s) it depends on: {late}",
                    )
                )
                continue
            completed += 1
        held = 0
        for assertion in expected.final_state:
            problem = check_assertion(assertion, trace.final_state)
            if problem is None:
                held += 1
            else:
                findings.append(finding("state_mismatch", f"final state {problem}"))
        total = len(expected.steps) + len(expected.final_state)
        return verdict(
            self.name,
            findings,
            score=ratio(completed + held, total),
            details={
                "steps": len(expected.steps),
                "steps_completed": completed,
                "state_assertions": len(expected.final_state),
                "state_assertions_held": held,
            },
        )
