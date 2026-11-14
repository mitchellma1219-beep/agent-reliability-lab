"""Tables for the dashboard. Pure functions over runs; no Streamlit here."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from ..demo import generate_demo_runs
from ..metrics import RunMetrics, compute_metrics
from ..models import CaseResult, LlmCall, RunResult, ToolCall, Trace
from ..regression import RegressionReport
from ..report import inline, plain
from ..storage import RunStore

DEMO_SOURCE = (
    "Demo data: the release history of a fictional agent, generated on this machine from "
    "the bundled suites and simulated agents. No real product, no real measurements."
)


@lru_cache(maxsize=1)
def _demo_runs() -> tuple[RunResult, ...]:
    return tuple(generate_demo_runs())


def load_runs(runs_dir: str | None) -> tuple[list[RunResult], list[str], str]:
    """Runs to show, problems met while reading them, and a description of the source."""
    if not runs_dir:
        return list(_demo_runs()), [], DEMO_SOURCE
    store = RunStore(runs_dir)
    runs: list[RunResult] = []
    problems: list[str] = []
    for name in store.names():
        try:
            runs.append(store.load(name))
        except Exception as exc:
            problems.append(plain(exc, 300))
    runs.sort(key=lambda run: (run.created_at, run.run_id))
    return runs, problems, f"Stored runs from {inline(runs_dir, 200)}"


def run_label(run: RunResult) -> str:
    return plain(
        f"{run.agent_name} {run.agent_version} / {run.suite_name} / {run.created_at[:10]} / "
        f"{run.run_id}",
        160,
    )


def runs_frame(runs: list[RunResult], metrics: list[RunMetrics]) -> pd.DataFrame:
    rows = [
        {
            "run": plain(run.run_id, 100),
            "agent": plain(run.agent_name, 60),
            "version": plain(run.agent_version, 30),
            "suite": plain(run.suite_name, 60),
            "created": run.created_at,
            "cases": m.cases,
            "trials": m.trials,
            "pass rate": m.pass_rate.rate,
            "task success": m.task_success_rate.rate,
            "tool-call accuracy": m.tool_call_accuracy.rate,
            "p95 latency (ms)": m.latency.p95_ms,
            "est. cost (USD)": m.estimated_cost_usd,
            "flaky cases": len(m.flaky_cases),
        }
        for run, m in zip(runs, metrics, strict=True)
    ]
    return pd.DataFrame(rows)


def categories_frame(metrics: RunMetrics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": item.category,
                "cases": item.cases,
                "passed": item.pass_rate.successes,
                "pass rate": item.pass_rate.rate,
                "task success": item.task_success_rate.rate,
                "mean latency (ms)": item.mean_latency_ms,
                "mean est. cost (USD)": item.mean_cost_usd,
            }
            for item in metrics.categories
        ]
    )


def evaluators_frame(metrics: RunMetrics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evaluator": item.evaluator,
                "applicable": item.applicable,
                "passed": item.passed,
                "failed": item.failed,
                "pass rate": item.pass_rate,
            }
            for item in metrics.evaluators
        ]
    )


def taxonomy_frame(metrics: RunMetrics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"failure category": name, "case-trials": count}
            for name, count in metrics.taxonomy.items()
        ],
        columns=["failure category", "case-trials"],
    )


def cases_frame(metrics: RunMetrics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case": plain(item.case_id, 100),
                "category": item.category,
                "outcome": item.outcome,
                "passes": f"{item.passes}/{item.trials}",
                "mean latency (ms)": item.mean_latency_ms,
                "mean est. cost (USD)": item.mean_cost_usd,
                "failing checks": ", ".join(item.failed_evaluators),
            }
            for item in metrics.case_stats
        ]
    )


def failures_frame(run: RunResult) -> pd.DataFrame:
    rows = []
    for result in run.results:
        if result.passed:
            continue
        first = next((f for e in result.evaluations for f in e.failures()), None)
        rows.append(
            {
                "case": plain(result.case_id, 100),
                "trial": result.trial,
                "category": result.category,
                "failing checks": ", ".join(result.failed_evaluators()),
                "failure categories": ", ".join(result.failure_categories),
                "first finding": plain(first.message, 200) if first else "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "case",
            "trial",
            "category",
            "failing checks",
            "failure categories",
            "first finding",
        ],
    )


def evaluations_frame(result: CaseResult) -> pd.DataFrame:
    rows = []
    for evaluation in result.evaluations:
        findings = "; ".join(f"{f.code}: {plain(f.message, 160)}" for f in evaluation.findings)
        rows.append(
            {
                "evaluator": evaluation.evaluator,
                "result": "pass" if evaluation.passed else "FAIL",
                "score": evaluation.score,
                "findings": findings,
            }
        )
    return pd.DataFrame(rows, columns=["evaluator", "result", "score", "findings"])


def _short(value: Any, limit: int = 120) -> str:
    return plain(json.dumps(value, ensure_ascii=False, default=str), limit)


def steps_frame(trace: Trace) -> pd.DataFrame:
    """One row per recorded step, in order, for the trace viewer and the timeline."""
    rows = []
    for step in trace.steps():
        if isinstance(step, ToolCall):
            outcome = "ok" if step.ok else f"failed: {step.error.code if step.error else 'error'}"
            rows.append(
                {
                    "step": step.index,
                    "kind": "tool call",
                    "name": plain(step.tool, 60),
                    "outcome": outcome,
                    "start (ms)": step.started_ms,
                    "duration (ms)": step.latency_ms,
                    "detail": _short(step.arguments),
                    "injected fault": step.fault or "",
                }
            )
        elif isinstance(step, LlmCall):
            rows.append(
                {
                    "step": step.index,
                    "kind": "model call",
                    "name": plain(step.model, 60),
                    "outcome": "ok",
                    "start (ms)": step.started_ms,
                    "duration (ms)": step.latency_ms,
                    "detail": f"{step.input_tokens:,} tokens in, {step.output_tokens:,} out"
                    + (f" ({plain(step.note, 60)})" if step.note else ""),
                    "injected fault": "",
                }
            )
        else:
            rows.append(
                {
                    "step": step.index,
                    "kind": "wait",
                    "name": "wait",
                    "outcome": "ok",
                    "start (ms)": step.started_ms,
                    "duration (ms)": step.ms,
                    "detail": plain(step.reason, 120),
                    "injected fault": "",
                }
            )
    columns = [
        "step",
        "kind",
        "name",
        "outcome",
        "start (ms)",
        "duration (ms)",
        "detail",
        "injected fault",
    ]
    return pd.DataFrame(rows, columns=columns)


def series_keys(runs: list[RunResult]) -> list[tuple[str, str]]:
    """Distinct (agent, suite) pairs, most runs first: each is one history to chart."""
    counts: dict[tuple[str, str], int] = {}
    for run in runs:
        key = (run.agent_name, run.suite_name)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts, key=lambda key: (-counts[key], key))


def series_runs(runs: list[RunResult], key: tuple[str, str]) -> list[RunResult]:
    chosen = [run for run in runs if (run.agent_name, run.suite_name) == key]
    return sorted(chosen, key=lambda run: (run.created_at, run.run_id))


def trend_frame(runs: list[RunResult]) -> pd.DataFrame:
    """One row per run of a single (agent, suite) history, oldest first."""
    rows = []
    for run in runs:
        m = compute_metrics(run)
        rows.append(
            {
                "version": plain(run.agent_version, 30),
                "created": run.created_at,
                "pass rate": m.pass_rate.rate,
                "task success": m.task_success_rate.rate,
                "tool-call accuracy": m.tool_call_accuracy.rate,
                "p95 latency (ms)": m.latency.p95_ms,
                "mean est. cost per case-trial (USD)": m.mean_cost_usd_per_trial,
                "flaky cases": len(m.flaky_cases),
                "run": plain(run.run_id, 100),
            }
        )
    return pd.DataFrame(rows)


def comparison_frame(baseline: RunMetrics, candidate: RunMetrics) -> pd.DataFrame:
    """Pass rate per category for two runs, for categories present in both."""
    before = {item.category: item for item in baseline.categories}
    rows = []
    for item in candidate.categories:
        if item.category in before:
            rows.append(
                {
                    "category": item.category,
                    "baseline": before[item.category].pass_rate.rate,
                    "candidate": item.pass_rate.rate,
                    "cases": item.cases,
                }
            )
    return pd.DataFrame(rows, columns=["category", "baseline", "candidate", "cases"])


def changes_frame(report: RegressionReport) -> pd.DataFrame:
    rows = []
    for label, changes in (
        ("newly failing", report.newly_failing),
        ("fixed", report.fixed),
        ("flaky", report.flaky),
    ):
        for change in changes:
            rows.append(
                {
                    "change": label,
                    "case": plain(change.case_id, 100),
                    "category": change.category,
                    "baseline": f"{change.baseline} ({change.baseline_passes})",
                    "candidate": f"{change.candidate} ({change.candidate_passes})",
                    "failing checks": ", ".join(change.failed_evaluators),
                }
            )
    return pd.DataFrame(
        rows, columns=["change", "case", "category", "baseline", "candidate", "failing checks"]
    )
