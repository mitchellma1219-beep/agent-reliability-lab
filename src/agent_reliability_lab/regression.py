"""Regression testing between two runs of an agent (usually two versions).

Cases are paired by id. A case *passes* a run only if every trial passed, *fails* if every
trial failed and is *flaky* otherwise.

* Reliability: the exact McNemar test on the paired pass / not-pass outcomes. Only cases
  whose outcome differs carry information; identical runs have none and get ``p = 1``.
* Cost and latency: the mean paired difference per case with a seeded percentile bootstrap
  interval. A change is reported only if the interval excludes zero *and* the relative
  change is at least ``tolerance``.

The verdict is deliberately conservative in one direction: any case that used to pass
consistently and no longer does is surfaced as ``needs_review`` even when the suite is too
small for the change in pass rate to be statistically significant.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .errors import ConfigError
from .metrics import Rate, case_outcome
from .models import CaseResult, Model, RunResult
from .stats import bootstrap_mean_difference, mcnemar_exact

Verdict = Literal["regression", "needs_review", "improved", "no_change"]


class RunRef(Model):
    run_id: str
    suite: str
    suite_version: str
    agent: str
    version: str
    created_at: str
    seed: int
    trials: int


class CaseChange(Model):
    case_id: str
    category: str
    baseline: str
    candidate: str
    baseline_passes: str
    candidate_passes: str
    failed_evaluators: list[str] = Field(default_factory=list)


class MetricChange(Model):
    """Change in a per-case mean between the two runs (candidate minus baseline)."""

    metric: str
    unit: str
    baseline_mean: float
    candidate_mean: float
    difference: float
    relative: float | None = None
    low: float
    high: float
    changed: bool
    direction: Literal["higher", "lower", "unchanged"]


class RegressionReport(Model):
    baseline: RunRef
    candidate: RunRef
    alpha: float
    tolerance: float
    cases_compared: int
    only_in_baseline: list[str] = Field(default_factory=list)
    only_in_candidate: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    baseline_pass_rate: Rate
    candidate_pass_rate: Rate
    newly_failing: list[CaseChange] = Field(default_factory=list)
    fixed: list[CaseChange] = Field(default_factory=list)
    flaky: list[CaseChange] = Field(default_factory=list)
    still_failing: list[str] = Field(default_factory=list)
    stopped_passing: int = 0
    started_passing: int = 0
    p_value: float = 1.0
    latency: MetricChange | None = None
    cost: MetricChange | None = None
    verdict: Verdict = "no_change"
    reasons: list[str] = Field(default_factory=list)


def _ref(run: RunResult) -> RunRef:
    return RunRef(
        run_id=run.run_id,
        suite=run.suite_name,
        suite_version=run.suite_version,
        agent=run.agent_name,
        version=run.agent_version,
        created_at=run.created_at,
        seed=run.config.seed,
        trials=run.config.trials,
    )


def _passes(results: list[CaseResult]) -> str:
    return f"{sum(1 for result in results if result.passed)}/{len(results)}"


def _metric(
    name: str,
    unit: str,
    baseline: list[float],
    candidate: list[float],
    tolerance: float,
    resamples: int,
    seed: int,
) -> MetricChange:
    result = bootstrap_mean_difference(baseline, candidate, resamples=resamples, seed=seed)
    baseline_mean = sum(baseline) / len(baseline)
    candidate_mean = sum(candidate) / len(candidate)
    relative = result.estimate / baseline_mean if baseline_mean > 0 else None
    material = relative is None or abs(relative) >= tolerance
    changed = result.interval.excludes_zero() and material
    direction: Literal["higher", "lower", "unchanged"] = "unchanged"
    if changed:
        direction = "higher" if result.estimate > 0 else "lower"
    return MetricChange(
        metric=name,
        unit=unit,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        difference=result.estimate,
        relative=relative,
        low=result.interval.low,
        high=result.interval.high,
        changed=changed,
        direction=direction,
    )


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    alpha: float = 0.05,
    tolerance: float = 0.10,
    resamples: int = 2000,
    seed: int = 0,
) -> RegressionReport:
    """Compare a candidate run with a baseline run of the same suite."""
    if not 0.0 < alpha < 1.0:
        raise ConfigError("alpha must be between 0 and 1")
    if tolerance < 0:
        raise ConfigError("tolerance must not be negative")
    before, after = baseline.by_case(), candidate.by_case()
    shared = [case_id for case_id in before if case_id in after]
    if not shared:
        raise ConfigError("the two runs have no cases in common; nothing can be compared")

    warnings: list[str] = []
    if baseline.suite_name != candidate.suite_name:
        warnings.append(
            f"different suites ({baseline.suite_name} vs {candidate.suite_name}); "
            "only cases with the same id are compared"
        )
    elif baseline.suite_version != candidate.suite_version:
        warnings.append("the suite version changed between the runs")
    if baseline.config.seed != candidate.config.seed:
        warnings.append("the runs used different seeds, so injected faults may differ")
    if baseline.config.trials != candidate.config.trials:
        warnings.append("the runs used a different number of trials")
    if baseline.config.fault_profile != candidate.config.fault_profile:
        warnings.append("the runs used different fault profiles")
    if baseline.config.prices != candidate.config.prices:
        warnings.append("the runs used different price tables; cost is not comparable")

    report = RegressionReport(
        baseline=_ref(baseline),
        candidate=_ref(candidate),
        alpha=alpha,
        tolerance=tolerance,
        cases_compared=len(shared),
        only_in_baseline=[case_id for case_id in before if case_id not in after],
        only_in_candidate=[case_id for case_id in after if case_id not in before],
        warnings=warnings,
        baseline_pass_rate=Rate.of(
            sum(case_outcome(before[case_id]) == "pass" for case_id in shared), len(shared)
        ),
        candidate_pass_rate=Rate.of(
            sum(case_outcome(after[case_id]) == "pass" for case_id in shared), len(shared)
        ),
    )
    if report.only_in_baseline or report.only_in_candidate:
        report.warnings.append("some cases exist in only one run and were left out")

    for case_id in shared:
        old, new = case_outcome(before[case_id]), case_outcome(after[case_id])
        change = CaseChange(
            case_id=case_id,
            category=after[case_id][0].category,
            baseline=old,
            candidate=new,
            baseline_passes=_passes(before[case_id]),
            candidate_passes=_passes(after[case_id]),
            failed_evaluators=sorted(
                {name for result in after[case_id] for name in result.failed_evaluators()}
            ),
        )
        report.stopped_passing += old == "pass" and new != "pass"
        report.started_passing += old != "pass" and new == "pass"
        if old == "pass" and new == "fail":
            report.newly_failing.append(change)
        elif old == "fail" and new == "pass":
            report.fixed.append(change)
        elif "flaky" in (old, new):
            report.flaky.append(change)
        elif old == "fail" and new == "fail":
            report.still_failing.append(case_id)

    report.p_value = mcnemar_exact(report.stopped_passing, report.started_passing)

    def per_case(run_cases: dict[str, list[CaseResult]], pick: str) -> list[float] | None:
        values: list[float] = []
        for case_id in shared:
            trials = run_cases[case_id]
            if pick == "latency":
                values.append(sum(r.trace.latency_ms for r in trials) / len(trials))
            else:
                costs = [r.estimated_cost_usd for r in trials]
                if any(cost is None for cost in costs):
                    return None
                values.append(sum(cost for cost in costs if cost is not None) / len(trials))
        return values

    latency_before, latency_after = per_case(before, "latency"), per_case(after, "latency")
    if latency_before is not None and latency_after is not None:
        report.latency = _metric(
            "latency", "ms", latency_before, latency_after, tolerance, resamples, seed
        )
    cost_before, cost_after = per_case(before, "cost"), per_case(after, "cost")
    if cost_before is not None and cost_after is not None:
        report.cost = _metric(
            "estimated cost", "USD", cost_before, cost_after, tolerance, resamples, seed
        )
    else:
        report.warnings.append("cost could not be compared: some model usage has no price")

    _decide(report)
    return report


def _decide(report: RegressionReport) -> None:
    b, c = report.stopped_passing, report.started_passing
    reasons: list[str] = []
    significant = report.p_value < report.alpha
    slower = report.latency is not None and report.latency.direction == "higher"
    faster = report.latency is not None and report.latency.direction == "lower"
    dearer = report.cost is not None and report.cost.direction == "higher"
    cheaper = report.cost is not None and report.cost.direction == "lower"

    if b:
        reasons.append(
            f"{b} case(s) that passed consistently no longer do "
            f"({len(report.newly_failing)} now fail every trial)"
        )
    if c:
        reasons.append(f"{c} case(s) now pass consistently that did not before")
    if b or c:
        reasons.append(
            f"exact McNemar test on {b} vs {c} changed cases: p = {report.p_value:.4f} "
            f"({'significant' if significant else 'not significant'} at alpha = {report.alpha:g})"
        )
    for change, worse in ((report.latency, slower), (report.cost, dearer)):
        if change is not None and change.changed:
            share = f"{change.relative:+.1%}" if change.relative is not None else "n/a"
            reasons.append(
                f"mean {change.metric} per case is {'higher' if worse else 'lower'}: {share} "
                f"(95% bootstrap interval of the difference {change.low:.6g} to "
                f"{change.high:.6g} {change.unit})"
            )

    if significant and b > c:
        report.verdict = "regression"
    elif b or slower or dearer:
        report.verdict = "needs_review"
        if not b and c:
            reasons.append(
                "reliability improved, but cost or latency rose: review whether the trade-off "
                "is acceptable"
            )
    elif c or faster or cheaper:
        report.verdict = "improved"
    else:
        report.verdict = "no_change"
        reasons.append("no case changed outcome and neither cost nor latency moved materially")
    report.reasons = reasons
