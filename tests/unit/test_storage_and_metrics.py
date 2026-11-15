"""Run storage with safe paths, and the observability metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_reliability_lab import faulty_agent, reference_agent, run_suite
from agent_reliability_lab.errors import StorageError
from agent_reliability_lab.metrics import Rate, case_outcome, compute_metrics
from agent_reliability_lab.models import RunConfig, RunResult, Suite
from agent_reliability_lab.storage import (
    RunStore,
    load_run_file,
    parse_run,
    resolve_run,
    validate_run_name,
)
from helpers import case_data, make_suite


@pytest.fixture(scope="module")
def run(core_suite: Suite) -> RunResult:
    return run_suite(core_suite, reference_agent(), run_id="reference-core")


# --------------------------------------------------------------------------- storage


@pytest.mark.parametrize(
    "name", ["run1", "core-reference-1.0.0-20261114T080000Z", "a.b_c-d", "X" * 100]
)
def test_valid_run_names(name: str) -> None:
    assert validate_run_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "../evil",
        "..",
        "a/b",
        "a\\b",
        "/etc/passwd",
        ".hidden",
        "-flag",
        "a..b",
        "trailing.",
        "name with space",
        "semi;colon",
        "new\nline",
        "nul\x00byte",
        "x" * 101,
        "CON",
        "nul.json",
        "com1",
        "~root",
        "$(whoami)",
        "run*",
        "über",
        None,
        5,
        b"bytes",
    ],
)
def test_unsafe_run_names_are_refused(name: object) -> None:
    with pytest.raises(StorageError, match="invalid run name"):
        validate_run_name(name)


def test_save_load_round_trip_is_exact(tmp_path: Path, run: RunResult) -> None:
    store = RunStore(tmp_path / "runs")
    path = store.save(run)
    assert path == tmp_path / "runs" / "reference-core.json" and store.exists("reference-core")
    loaded = store.load("reference-core")
    assert loaded == run and loaded.fingerprint() == run.fingerprint()
    assert [entry.name for entry in (tmp_path / "runs").iterdir()] == ["reference-core.json"]


def test_save_refuses_to_overwrite_unless_asked(tmp_path: Path, run: RunResult) -> None:
    store = RunStore(tmp_path)
    store.save(run)
    with pytest.raises(StorageError, match="already exists"):
        store.save(run)
    changed = run.model_copy(update={"notes": "second"})
    store.save(changed, overwrite=True)
    assert store.load(run.run_id).notes == "second"


def test_symlinks_are_not_followed(tmp_path: Path, run: RunResult) -> None:
    store = RunStore(tmp_path / "runs")
    store.save(run)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = tmp_path / "runs" / "linked.json"
    link.symlink_to(outside)
    with pytest.raises(StorageError, match="outside"):
        store.load("linked")
    with pytest.raises(StorageError, match="outside"):
        store.save(run.model_copy(update={"run_id": "linked"}), overwrite=True)
    assert outside.read_text(encoding="utf-8") == "{}"
    assert store.names() == ["reference-core"], "symlinks are not listed"
    with pytest.raises(StorageError, match="symbolic link"):
        load_run_file(link)


def test_failed_write_leaves_no_partial_file(
    tmp_path: Path, run: RunResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        store.save(run)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "content, message",
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "not a stored run"),
        ('{"schema_version": 99}', "schema version 99 is not supported"),
        ('{"run_id": "x"}', "schema version None"),
        ('{"schema_version": 1, "run_id": "x"}', "not a valid run"),
        ("[" * 100_000, "not valid JSON"),
    ],
)
def test_unreadable_run_files_are_storage_errors(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(StorageError, match=message):
        load_run_file(path)
    with pytest.raises(StorageError, match=message):
        parse_run(content, "inline")


def test_missing_and_binary_files(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="no stored run named"):
        RunStore(tmp_path).load("absent")
    with pytest.raises(StorageError, match="no such run file"):
        load_run_file(tmp_path / "absent.json")
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(StorageError, match="UTF-8"):
        load_run_file(binary)


def test_listing_skips_bad_files_and_reports_them(
    tmp_path: Path, run: RunResult, core_suite: Suite
) -> None:
    store = RunStore(tmp_path)
    assert store.list() == ([], []) and store.names() == []
    store.save(run)
    second = run_suite(
        core_suite, faulty_agent("wrong_tool"), RunConfig(trials=2), run_id="faulty-core"
    )
    store.save(second)
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "bad name.json").write_text("{}", encoding="utf-8")
    infos, problems = store.list()
    assert sorted(info.name for info in infos) == ["faulty-core", "reference-core"]
    assert len(problems) == 1 and "broken.json" in problems[0]
    by_name = {info.name: info for info in infos}
    assert by_name["reference-core"].pass_rate == 1.0 and by_name["reference-core"].total == 17
    assert by_name["faulty-core"].trials == 2 and by_name["faulty-core"].passed == 0


def test_resolve_run_accepts_a_name_or_an_explicit_path(tmp_path: Path, run: RunResult) -> None:
    store = RunStore(tmp_path)
    path = store.save(run)
    assert resolve_run("reference-core", store) == run
    assert resolve_run(str(path), store) == run
    with pytest.raises(StorageError, match="invalid run name"):
        resolve_run("bad name", store)
    with pytest.raises(StorageError, match="no such run file"):
        resolve_run("../../etc/passwd", store)


def test_stored_file_is_plain_json(tmp_path: Path, run: RunResult) -> None:
    path = RunStore(tmp_path).save(run)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["run_id"] == "reference-core"
    assert len(data["results"]) == 17 and "trace" in data["results"][0]


# --------------------------------------------------------------------------- metrics


def test_rate_carries_a_wilson_interval() -> None:
    rate = Rate.of(58, 60)
    assert rate.rate == pytest.approx(58 / 60)
    assert (rate.low, rate.high) == pytest.approx((0.886362, 0.990811), abs=1e-6)
    assert Rate.of(0, 0) == Rate() and Rate.of(0, 0).rate is None


def test_metrics_for_a_clean_run(run: RunResult) -> None:
    metrics = compute_metrics(run)
    assert (metrics.cases, metrics.trials, metrics.case_trials) == (17, 1, 17)
    assert metrics.pass_rate.rate == 1.0 and metrics.task_success_rate.rate == 1.0
    assert metrics.tool_call_accuracy.total == 17 and metrics.tool_call_recall == 1.0
    assert metrics.failure_rate == 0.0 and metrics.error_rate.successes == 0
    assert metrics.taxonomy == {} and metrics.flaky_cases == []
    assert metrics.tool_calls == sum(len(r.trace.tool_calls) for r in run.results) == 27
    assert metrics.retries == 0 and metrics.retry_rate == 0.0 and metrics.injected_faults == 0
    assert metrics.input_tokens == sum(r.trace.usage.input_tokens for r in run.results)
    assert metrics.estimated_cost_usd == pytest.approx(
        sum(r.estimated_cost_usd or 0 for r in run.results)
    )
    assert metrics.price_table == "illustrative" and metrics.unpriced_trials == 0
    latencies = sorted(r.trace.latency_ms for r in run.results)
    assert metrics.latency.max_ms == latencies[-1] and metrics.latency.p50_ms == latencies[8]
    assert (
        metrics.latency.p50_ms
        <= metrics.latency.p90_ms
        <= metrics.latency.p95_ms
        <= metrics.latency.p99_ms
    )
    assert [item.category for item in metrics.categories] == [
        "lookup",
        "calculation",
        "grounded_qa",
        "structured_output",
        "instruction",
        "workflow",
    ]
    assert sum(item.cases for item in metrics.categories) == 17
    by_name = {item.evaluator: item for item in metrics.evaluators}
    assert by_name["task_success"].applicable == 17 and by_name["output_schema"].applicable == 2
    assert all(item.failed == 0 for item in metrics.evaluators)


def test_metrics_for_a_faulty_run_count_failures_by_category(core_suite: Suite) -> None:
    faulty = run_suite(core_suite, faulty_agent("invalid_json"))
    metrics = compute_metrics(faulty)
    assert metrics.pass_rate.successes == 15 and metrics.failure_rate == pytest.approx(2 / 17)
    assert metrics.taxonomy == {"schema_violation": 2, "task_failure": 2}
    structured = next(item for item in metrics.categories if item.category == "structured_output")
    assert structured.pass_rate.rate == 0.0 and structured.cases == 2
    failing = [stat for stat in metrics.case_stats if stat.outcome == "fail"]
    assert [stat.case_id for stat in failing] == [
        "core-json-order-summary",
        "core-json-stock-report",
    ]
    assert failing[0].failed_evaluators == ["output_schema", "task_success"]


def test_flaky_cases_and_retry_statistics() -> None:
    suite = make_suite(
        case_data(
            id="flaky-lookup",
            input="What is the status of order A-1001?",
            faults=[{"kind": "transient_error", "tool": "get_order", "times": 1}],
            sim={
                "steps": [{"id": "order", "tool": "get_order", "args": {"order_id": "A-1001"}}],
                "answer": "Order {order.id} is {order.status}.",
            },
            expected={"answer": {"contains_all": ["shipped"]}},
        )
    )
    flaky = run_suite(
        suite, faulty_agent("no_retry", activation_rate=0.5), RunConfig(trials=12, seed=3)
    )
    metrics = compute_metrics(flaky)
    stat = metrics.case_stats[0]
    assert (
        0 < stat.passes < 12 and stat.outcome == "flaky" and metrics.flaky_cases == ["flaky-lookup"]
    )
    assert metrics.pass_rate.rate == 0.0, "a flaky case does not count as passed"
    assert metrics.trial_pass_rate.successes == stat.passes
    assert metrics.injected_faults == 12
    assert metrics.retries == stat.passes == metrics.trials_with_retries
    assert metrics.retry_rate == pytest.approx(stat.passes / metrics.tool_calls)


def test_case_outcome() -> None:
    suite = make_suite(case_data())
    passing = run_suite(suite, reference_agent(), RunConfig(trials=2)).results
    assert all(not result.passed for result in passing), (
        "no sim block: the simulated agent cannot run it"
    )
    assert case_outcome(passing) == "fail"


def test_metrics_with_an_unpriced_model(core_suite: Suite) -> None:
    from agent_reliability_lab.models import PriceTable

    run = run_suite(core_suite, reference_agent(), RunConfig(prices=PriceTable(name="empty")))
    metrics = compute_metrics(run)
    assert metrics.estimated_cost_usd is None and metrics.mean_cost_usd_per_trial is None
    assert metrics.unpriced_trials == 17
    assert all(category.mean_cost_usd is None for category in metrics.categories)
