"""Instruction following: every effective machine-checkable rule must hold for the output."""

from __future__ import annotations

from pydantic import JsonValue

from ..evidence import known_sources, sink_text
from ..instructions import check_rule, resolve_rules
from ..models import Case, Evaluation, Finding, Rule, Trace
from .base import EvalContext, finding, ratio, verdict


def _relaxed(rule: Rule) -> Rule:
    """Rules about the substance of an answer do not apply to a question, refusal or failure."""
    return rule.model_copy(update={"cite_sources": False, "min_words": None})


class InstructionFollowing:
    """Checks the rules attached to the case's instructions.

    System instructions outrank user instructions; a user instruction that contradicts a
    system instruction is reported as overridden and not checked. When the agent did not
    complete the task (it asked, refused, abstained or failed), ``cite_sources`` and
    ``min_words`` are not applied; every other rule still is.
    """

    name = "instruction_following"

    def applies(self, case: Case, trace: Trace) -> bool:
        return case.has_rules()

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        resolved = resolve_rules(case.instructions)
        sources = known_sources(trace, ctx.context)
        sinks = sink_text(trace)
        findings: list[Finding] = []
        violated: list[JsonValue] = []
        rules = resolved.rules()
        for instruction_id, rule in rules:
            active = rule if trace.status == "completed" else _relaxed(rule)
            problems = check_rule(active, trace.output, sink_text=sinks, known_sources=sources)
            if problems:
                violated.append(instruction_id)
            findings.extend(
                finding("instruction_violated", f"instruction {instruction_id!r}: {problem}")
                for problem in problems
            )
        for instruction, reason in resolved.overridden:
            findings.append(
                finding(
                    "instruction_overridden", f"{instruction.id!r} not checked: {reason}", "info"
                )
            )
        return verdict(
            self.name,
            findings,
            score=ratio(len(rules) - len(violated), len(rules)),
            details={
                "rules_checked": len(rules),
                "rules_violated": violated,
                "overridden": [item.id for item, _ in resolved.overridden],
            },
        )
