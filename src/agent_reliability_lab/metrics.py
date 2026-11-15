"""Observability: turn a stored run into rates, percentiles, usage, cost and breakdowns.

Definitions (also printed in reports):

* **pass rate** counts a case as passed only if *every* trial passed *every* applicable
  evaluator. Its Wilson interval treats the cases as a sample of the tasks the agent may
  meet, which is an assumption, not a fact about any particular suite.
* **task success rate** uses only the ``task_success`` evaluator.
* **tool-call accuracy** is the share of case-trials whose tool calls matched the
  expectation; precision and recall are pooled over all calls.
* **latency** is accounted time (simulated tool latency, reported model latency, waits).
* **cost** is an estimate: reported tokens multiplied by the run's price table.
"""

from __future__ import annotations

from collections import Counter

from pydantic import Field, JsonValue

from .models import CaseResult, Model, RunResult
from .stats import percentile, wilson_interval


class Rate(Model):
    """A proportion with its Wilson 95% interval."""

    successes: int = 0
    total: int = 0
    rate: float | None = None
    low: float | None = None
    high: float | None = None

    @classmethod
    def of(cls, successes: int, total: int) -> Rate:
        if total == 0:
            return cls()
        interval = wilson_interval(successes, total)
        return cls(
            successes=successes,
            total=total,
            rate=successes / total,
            low=interval.low,
            high=interval.high,
        )


class LatencyStats(Model):
    n: int = 0
    mean_ms: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    max_ms: float | None = None

    @classmethod
    def of(cls, values: list[float]) -> LatencyStats:
        if not values:
            return cls()
        return cls(
            n=len(values),
            mean_ms=sum(values) / len(values),
            p50_ms=percentile(values, 50),
            p90_ms=percentile(values, 90),
            p95_ms=percentile(values, 95),
            p99_ms=percentile(values, 99),
            max_ms=max(values),
        )


class EvaluatorStat(Model):
    evaluator: str
    applicable: int
    passed: int
    failed: int
    pass_rate: float


class CategoryStat(Model):
    category: str
    cases: int
    pass_rate: Rate
    task_success_rate: Rate
    mean_latency_ms: float | None = None
    mean_cost_usd: float | None = None


class CaseStat(Model):
    case_id: str
    category: str
    trials: int
    passes: int
    outcome: str
    pass_rate: Rate
    mean_latency_ms: float
    mean_cost_usd: float | None = None
    failed_evaluators: list[str] = Field(default_factory=list)


class RunMetrics(Model):
    run_id: str
    suite: str
    agent: str
    version: str
    created_at: str
    cases: int
    trials: int
    case_trials: int
    pass_rate: Rate
    trial_pass_rate: Rate
    task_success_rate: Rate
    tool_call_accuracy: Rate
    tool_call_precision: float | None = None
    tool_call_recall: float | None = None
    failure_rate: float | None = None
    error_rate: Rate
    tool_calls: int = 0
    retries: int = 0
    retry_rate: float | None = None
    trials_with_retries: int = 0
    injected_faults: int = 0
    latency: LatencyStats
    input_tokens: int = 0
    output_tokens: int = 0
    mean_tokens_per_trial: float | None = None
    estimated_cost_usd: float | None = None
    mean_cost_usd_per_trial: float | None = None
    unpriced_trials: int = 0
    price_table: str = ""
    taxonomy: dict[str, int] = Field(default_factory=dict)
    evaluators: list[EvaluatorStat] = Field(default_factory=list)
    categories: list[CategoryStat] = Field(default_factory=list)
    case_stats: list[CaseStat] = Field(default_factory=list)
    flaky_cases: list[str] = Field(default_factory=list)


def _passed(result: CaseResult, evaluator: str) -> bool | None:
    evaluation = result.evaluation(evaluator)
    return None if evaluation is None else evaluation.passed


def _count(value: JsonValue) -> int:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def case_outcome(results: list[CaseResult]) -> str:
    """``pass`` if every trial passed, ``fail`` if none did, otherwise ``flaky``."""
    passes = sum(1 for result in results if result.passed)
    if passes == len(results):
        return "pass"
    return "fail" if passes == 0 else "flaky"


def compute_metrics(run: RunResult) -> RunMetrics:
    """All observability metrics for one run."""
    results = run.results
    by_case = run.by_case()

    case_stats: list[CaseStat] = []
    for case_id, trials in by_case.items():
        passes = sum(1 for result in trials if result.passed)
        costs = [r.estimated_cost_usd for r in trials if r.estimated_cost_usd is not None]
        case_stats.append(
            CaseStat(
                case_id=case_id,
                category=trials[0].category,
                trials=len(trials),
                passes=passes,
                outcome=case_outcome(trials),
                pass_rate=Rate.of(passes, len(trials)),
                mean_latency_ms=sum(r.trace.latency_ms for r in trials) / len(trials),
                mean_cost_usd=_mean(costs) if len(costs) == len(trials) else None,
                failed_evaluators=sorted({name for r in trials for name in r.failed_evaluators()}),
            )
        )

    task = [flag for flag in (_passed(r, "task_success") for r in results) if flag is not None]
    tools = [r.evaluation("tool_call_accuracy") for r in results]
    tool_evaluations = [item for item in tools if item is not None]

    def pooled(numerator: str, denominator: str) -> float | None:
        top = sum(_count(e.details.get(numerator)) for e in tool_evaluations)
        bottom = sum(_count(e.details.get(denominator)) for e in tool_evaluations)
        return top / bottom if bottom else None

    evaluator_counts: dict[str, list[int]] = {}
    for result in results:
        for evaluation in result.evaluations:
            counts = evaluator_counts.setdefault(evaluation.evaluator, [0, 0])
            counts[0] += 1
            counts[1] += evaluation.passed
    evaluators = [
        EvaluatorStat(
            evaluator=name,
            applicable=applicable,
            passed=passed,
            failed=applicable - passed,
            pass_rate=passed / applicable,
        )
        for name, (applicable, passed) in evaluator_counts.items()
    ]

    categories: list[CategoryStat] = []
    for category in dict.fromkeys(stat.category for stat in case_stats):
        members = [stat for stat in case_stats if stat.category == category]
        trials = [r for r in results if r.category == category]
        category_task = [f for f in (_passed(r, "task_success") for r in trials) if f is not None]
        costs = [r.estimated_cost_usd for r in trials if r.estimated_cost_usd is not None]
        categories.append(
            CategoryStat(
                category=category,
                cases=len(members),
                pass_rate=Rate.of(sum(s.outcome == "pass" for s in members), len(members)),
                task_success_rate=Rate.of(sum(category_task), len(category_task)),
                mean_latency_ms=_mean([r.trace.latency_ms for r in trials]),
                mean_cost_usd=_mean(costs) if len(costs) == len(trials) else None,
            )
        )

    tool_calls = sum(len(r.trace.tool_calls) for r in results)
    retries = sum(r.trace.retries for r in results)
    costs = [r.estimated_cost_usd for r in results if r.estimated_cost_usd is not None]
    priced = len(costs) == len(results)
    tokens_in = sum(r.trace.usage.input_tokens for r in results)
    tokens_out = sum(r.trace.usage.output_tokens for r in results)
    passed_cases = sum(1 for stat in case_stats if stat.outcome == "pass")
    passed_trials = sum(1 for r in results if r.passed)
    taxonomy = Counter(category for r in results for category in r.failure_categories)

    return RunMetrics(
        run_id=run.run_id,
        suite=run.suite_name,
        agent=run.agent_name,
        version=run.agent_version,
        created_at=run.created_at,
        cases=len(case_stats),
        trials=run.config.trials,
        case_trials=len(results),
        pass_rate=Rate.of(passed_cases, len(case_stats)),
        trial_pass_rate=Rate.of(passed_trials, len(results)),
        task_success_rate=Rate.of(sum(task), len(task)),
        tool_call_accuracy=Rate.of(
            sum(1 for item in tool_evaluations if item.passed), len(tool_evaluations)
        ),
        tool_call_precision=pooled("matched_calls", "successful_calls"),
        tool_call_recall=pooled("required_satisfied", "required"),
        failure_rate=1.0 - passed_trials / len(results) if results else None,
        error_rate=Rate.of(sum(1 for r in results if r.trace.errors), len(results)),
        tool_calls=tool_calls,
        retries=retries,
        retry_rate=retries / tool_calls if tool_calls else None,
        trials_with_retries=sum(1 for r in results if r.trace.retries),
        injected_faults=sum(1 for r in results for call in r.trace.tool_calls if call.fault),
        latency=LatencyStats.of([r.trace.latency_ms for r in results]),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        mean_tokens_per_trial=(tokens_in + tokens_out) / len(results) if results else None,
        estimated_cost_usd=sum(costs) if priced and results else None,
        mean_cost_usd_per_trial=_mean(costs) if priced else None,
        unpriced_trials=len(results) - len(costs),
        price_table=run.config.prices.name,
        taxonomy=dict(sorted(taxonomy.items(), key=lambda item: (-item[1], item[0]))),
        evaluators=evaluators,
        categories=categories,
        case_stats=case_stats,
        flaky_cases=[stat.case_id for stat in case_stats if stat.outcome == "flaky"],
    )
