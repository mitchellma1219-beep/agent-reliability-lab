"""The detection matrix: does the lab notice each kind of mistake, and only real mistakes?

For every failure mode a simulated agent with exactly that fault is run over the built-in
suites, next to the clean reference agent. The simulated agents report (as trace metadata)
in which cases the fault actually changed their behaviour, which gives ground truth to
compare the evaluators against:

* **detected**: the fault changed behaviour and at least one evaluator failed the case;
* **escape**: the fault changed behaviour and every evaluator passed;
* **false alarm**: an evaluator failed a case in which the fault did not change anything;
* **benign**: a known, explained deviation that does no harm and is rightly not flagged.

The matrix is a single-fault experiment on rule-based agents. It shows that the evaluators
and fixtures are wired correctly; it does not estimate detection rates for real models.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .agents.simulated import FAILURE_MODES, faulty_agent, reference_agent
from .models import RunConfig, RunResult
from .runner import run_suite
from .scenarios import builtin_suite_names, load_builtin_suite


@dataclass(frozen=True)
class Expectation:
    """What the matrix requires for one failure mode."""

    primary: tuple[str, ...]
    only: tuple[str, ...] = ()


EXPECTATIONS: dict[str, Expectation] = {
    "follows_injection": Expectation(("injection_resistance",)),
    "hallucinates": Expectation(("hallucination",)),
    "wrong_tool": Expectation(("tool_call_accuracy",)),
    "bad_arguments": Expectation(("tool_call_accuracy",)),
    "ignores_instruction": Expectation(("instruction_following",)),
    "invalid_json": Expectation(("output_schema",)),
    "no_retry": Expectation(("recovery",)),
    "unbounded_retry": Expectation(("recovery",)),
    "leaks_canary": Expectation(("canary_leak",)),
    "partial_completion": Expectation(("workflow",)),
    "slow": Expectation(("latency_budget",), only=("latency_budget",)),
    "expensive": Expectation(("cost_budget",), only=("cost_budget",)),
    "trusts_invalid_data": Expectation(("recovery", "task_success")),
    "guesses_when_ambiguous": Expectation(("task_success", "tool_call_accuracy")),
    "distracted": Expectation(("task_success",)),
}

# Deviations from the reference behaviour that are harmless, with the reason.
BENIGN: dict[tuple[str, str], str] = {
    ("no_retry", "rob-types-null-price"): (
        "the record is permanently invalid, so giving up after one attempt instead of three "
        "ends in the same explicit failure"
    ),
    ("no_retry", "rob-types-list-for-number"): (
        "the record is permanently invalid, so giving up after one attempt instead of three "
        "ends in the same explicit failure"
    ),
}

# The failure mode each scenario category is designed to expose.
CATEGORY_TARGETS: dict[str, str] = {
    "direct_injection": "follows_injection",
    "indirect_injection_document": "follows_injection",
    "indirect_injection_tool": "follows_injection",
    "conflicting_instructions": "ignores_instruction",
    "missing_context": "hallucinates",
    "ambiguous_request": "guesses_when_ambiguous",
    "long_context": "distracted",
    "unexpected_types": "trusts_invalid_data",
    "malformed_json": "trusts_invalid_data",
    "tool_outage": "hallucinates",
    "repeated_failures": "no_retry",
    "timeout": "no_retry",
    "rate_limit": "no_retry",
    "transient_error": "no_retry",
    "permanent_error": "unbounded_retry",
    "retry_exhaustion": "unbounded_retry",
    "dependency_failure": "unbounded_retry",
    "incorrect_response": "trusts_invalid_data",
    "invalid_output": "trusts_invalid_data",
    "partial_completion": "partial_completion",
}


@dataclass
class MatrixRow:
    mode: str
    description: str
    primary: tuple[str, ...]
    cases: int = 0
    triggered: int = 0
    detected: int = 0
    detected_by_primary: dict[str, int] = field(default_factory=dict)
    benign: list[str] = field(default_factory=list)
    escapes: list[str] = field(default_factory=list)
    false_alarms: list[str] = field(default_factory=list)
    flagged_by: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class DetectionMatrix:
    seed: int
    suites: list[str]
    reference_cases: int = 0
    reference_failures: list[str] = field(default_factory=list)
    rows: list[MatrixRow] = field(default_factory=list)
    category_coverage: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        uncovered = [name for name, (_, failed, _) in self.category_coverage.items() if not failed]
        return not self.reference_failures and not uncovered and all(row.ok for row in self.rows)

    def row(self, mode: str) -> MatrixRow:
        return next(row for row in self.rows if row.mode == mode)


def _triggered(run: RunResult, mode: str) -> set[str]:
    hit: set[str] = set()
    for result in run.results:
        modes = result.trace.agent_metadata.get("triggered")
        if isinstance(modes, list) and mode in modes:
            hit.add(result.case_id)
    return hit


def _row(mode: str, runs: list[RunResult]) -> MatrixRow:
    expectation = EXPECTATIONS[mode]
    row = MatrixRow(mode=mode, description=FAILURE_MODES[mode], primary=expectation.primary)
    flagged: Counter[str] = Counter()
    by_primary: Counter[str] = Counter()
    for run in runs:
        triggered = _triggered(run, mode)
        for result in run.results:
            row.cases += 1
            failed = result.failed_evaluators()
            flagged.update(failed)
            if result.case_id in triggered:
                row.triggered += 1
                if failed:
                    row.detected += 1
                    by_primary.update(name for name in expectation.primary if name in failed)
                elif (mode, result.case_id) in BENIGN:
                    row.benign.append(result.case_id)
                else:
                    row.escapes.append(result.case_id)
            elif failed:
                row.false_alarms.append(result.case_id)
    row.flagged_by = dict(sorted(flagged.items(), key=lambda item: (-item[1], item[0])))
    row.detected_by_primary = {name: by_primary.get(name, 0) for name in expectation.primary}

    if row.triggered == 0:
        row.problems.append("the fault never changed the agent's behaviour")
    for name, count in row.detected_by_primary.items():
        if count == 0:
            row.problems.append(f"{name} never flagged this fault")
    if row.escapes:
        row.problems.append(f"escaped detection in: {', '.join(row.escapes)}")
    if row.false_alarms:
        row.problems.append(f"flagged without the fault acting in: {', '.join(row.false_alarms)}")
    if expectation.only:
        others = sorted(set(flagged) - set(expectation.only))
        if others:
            row.problems.append(f"unrelated evaluators also failed: {', '.join(others)}")
        if row.detected != row.cases:
            row.problems.append("expected every case to be flagged")
    return row


def run_detection_matrix(
    suites: list[str] | None = None,
    seed: int = 0,
    modes: list[str] | None = None,
) -> DetectionMatrix:
    """Run the reference agent and every faulty agent over the built-in suites."""
    names = suites or builtin_suite_names()
    loaded = [load_builtin_suite(name) for name in names]
    config = RunConfig(seed=seed)
    matrix = DetectionMatrix(seed=seed, suites=list(names))

    for suite in loaded:
        run = run_suite(suite, reference_agent(), config, run_id=f"matrix-reference-{suite.name}")
        matrix.fingerprints[f"reference/{suite.name}"] = run.fingerprint()
        matrix.reference_cases += len(run.results)
        matrix.reference_failures += [
            f"{suite.name}/{result.case_id}: {', '.join(result.failed_evaluators())}"
            for result in run.results
            if not result.passed
        ]

    failures: dict[str, dict[str, tuple[int, int]]] = {}
    for mode in modes or list(FAILURE_MODES):
        runs = []
        for suite in loaded:
            run = run_suite(suite, faulty_agent(mode), config, run_id=f"matrix-{suite.name}")
            matrix.fingerprints[f"{mode}/{suite.name}"] = run.fingerprint()
            runs.append(run)
            per_category = failures.setdefault(mode, {})
            for result in run.results:
                failed, total = per_category.get(result.category, (0, 0))
                per_category[result.category] = (failed + (not result.passed), total + 1)
        matrix.rows.append(_row(mode, runs))

    categories = {case.category for suite in loaded for case in suite.cases}
    for category in sorted(categories & set(CATEGORY_TARGETS)):
        target = CATEGORY_TARGETS[category]
        if target in failures:
            failed, total = failures[target].get(category, (0, 0))
            matrix.category_coverage[category] = (target, failed, total)
    return matrix


def render_matrix(matrix: DetectionMatrix) -> str:
    """The matrix as Markdown."""
    lines = [
        f"Detection matrix (seed {matrix.seed}; suites: {', '.join(matrix.suites)})",
        "",
        f"Reference agent: {matrix.reference_cases - len(matrix.reference_failures)}"
        f"/{matrix.reference_cases} cases passed.",
        "",
        "| Failure mode | Expected evaluator(s) | Fault acted | Detected | By expected "
        "evaluator | Benign | Escapes | False alarms | Result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in matrix.rows:
        by_primary = ", ".join(f"{name} {count}" for name, count in row.detected_by_primary.items())
        lines.append(
            f"| {row.mode} | {', '.join(row.primary)} | {row.triggered}/{row.cases} | "
            f"{row.detected} | {by_primary} | {len(row.benign)} | {len(row.escapes)} | "
            f"{len(row.false_alarms)} | {'ok' if row.ok else 'PROBLEM'} |"
        )
    if matrix.category_coverage:
        lines += [
            "",
            "| Scenario category | Fault it is designed to expose | Cases failed by that fault |",
            "|---|---|---|",
        ]
        for category, (target, failed, total) in matrix.category_coverage.items():
            lines.append(f"| {category} | {target} | {failed}/{total} |")
    problems = [f"- {row.mode}: {problem}" for row in matrix.rows for problem in row.problems]
    problems += [f"- reference agent failed {item}" for item in matrix.reference_failures]
    lines += ["", "Result: " + ("all checks hold." if matrix.ok else "PROBLEMS FOUND"), *problems]
    return "\n".join(lines)
