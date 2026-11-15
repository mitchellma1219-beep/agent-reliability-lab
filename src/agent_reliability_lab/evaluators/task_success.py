"""Task success: did the agent end in an acceptable state with an acceptable answer?"""

from __future__ import annotations

import json
import re
from typing import Any

from ..matching import matches
from ..models import AnswerCheck, Case, Evaluation, Finding, Trace
from ..text import normalise_number
from .base import EvalContext, finding, ratio, verdict

_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")


def _norm(text: str, case_sensitive: bool) -> str:
    collapsed = " ".join(text.split())
    return collapsed if case_sensitive else collapsed.casefold()


def numbers_in(text: str) -> list[float]:
    """Numbers written in ``text``; thousands separators are understood."""
    found: list[float] = []
    for raw in _NUMBER.findall(text):
        canonical = normalise_number(raw)
        if canonical is not None:
            found.append(float(canonical))
    return found


def _json_path(payload: Any, dotted: str) -> tuple[bool, Any]:
    current = payload
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def check_answer(answer: AnswerCheck, output: str) -> tuple[list[Finding], int]:
    """Return the findings and the number of individual checks that were made."""
    findings: list[Finding] = []
    checks = 0
    text = _norm(output, answer.case_sensitive)
    for phrase in answer.contains_all:
        checks += 1
        if _norm(phrase, answer.case_sensitive) not in text:
            findings.append(finding("missing_text", f"answer does not contain {phrase!r}"))
    if answer.contains_any:
        checks += 1
        if not any(_norm(phrase, answer.case_sensitive) in text for phrase in answer.contains_any):
            findings.append(
                finding("missing_text", f"answer contains none of {answer.contains_any!r}")
            )
    for phrase in answer.not_contains:
        checks += 1
        if _norm(phrase, answer.case_sensitive) in text:
            findings.append(finding("forbidden_text", f"answer contains {phrase!r}"))
    flags = 0 if answer.case_sensitive else re.IGNORECASE
    for pattern in answer.regex:
        checks += 1
        if not re.search(pattern, output, flags):
            findings.append(finding("pattern_not_matched", f"answer does not match /{pattern}/"))
    if answer.equals is not None:
        checks += 1
        if _norm(answer.equals, answer.case_sensitive) != text:
            findings.append(finding("not_equal", "answer is not equal to the expected text"))
    if answer.numeric is not None:
        checks += 1
        target, tolerance = answer.numeric.value, answer.numeric.tolerance
        if not any(abs(value - target) <= tolerance for value in numbers_in(output)):
            findings.append(
                finding("number_not_found", f"answer does not state {target:g} (±{tolerance:g})")
            )
    if answer.json_fields is not None:
        try:
            payload = json.loads(output)
        except (ValueError, RecursionError):
            checks += len(answer.json_fields) or 1
            findings.append(finding("not_json", "answer is not valid JSON"))
        else:
            for path, expected in answer.json_fields.items():
                checks += 1
                present, actual = _json_path(payload, path)
                if not present or not matches(expected, actual):
                    shown = json.dumps(actual, default=str)[:80] if present else "missing"
                    findings.append(finding("json_field_mismatch", f"field {path!r} is {shown}"))
    return findings, checks


class TaskSuccess:
    """Checks the kind of answer (completed, asked, refused, failed safely) and its content.

    Any error raised by the harness for this run (crash, timeout, runaway loop, unusable
    output) also fails the task.
    """

    name = "task_success"

    def applies(self, case: Case, trace: Trace) -> bool:
        return True

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        findings: list[Finding] = [finding(error.kind, error.message) for error in trace.errors]
        checks = 1
        accepted = case.expected.statuses()
        if trace.status not in accepted:
            findings.append(
                finding(
                    "wrong_status",
                    f"expected the agent to end as {' or '.join(accepted)}, "
                    f"but it ended as {trace.status} ({trace.status_source})",
                )
            )
        if case.expected.answer is not None:
            answer_findings, answer_checks = check_answer(case.expected.answer, trace.output)
            findings.extend(answer_findings)
            checks += answer_checks
        failed = len({id(item) for item in findings})
        return verdict(
            self.name,
            findings,
            score=ratio(max(0, checks - failed), checks),
            details={"status": trace.status, "status_source": trace.status_source},
        )
