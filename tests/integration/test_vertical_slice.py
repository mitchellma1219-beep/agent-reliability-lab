"""The first end-to-end slice: suite -> agent -> sandbox -> trace -> evaluator -> result."""

from __future__ import annotations

from agent_reliability_lab import faulty_agent, reference_agent, run_suite
from agent_reliability_lab.models import RunConfig, Suite


def test_reference_agent_passes_the_core_suite(core_suite: Suite) -> None:
    run = run_suite(core_suite, reference_agent())
    failed = [(r.case_id, r.failed_evaluators()) for r in run.results if not r.passed]
    assert failed == []
    assert len(run.results) == len(core_suite.cases)


def test_wrong_tool_agent_is_caught_on_a_lookup(core_suite: Suite) -> None:
    run = run_suite(core_suite, faulty_agent("wrong_tool"))
    result = run.by_case()["core-order-status"][0]
    assert not result.passed
    assert "task_success" in result.failed_evaluators()
    assert [call.tool for call in result.trace.tool_calls] == ["get_customer"]


def test_trace_is_recorded_by_the_harness(core_suite: Suite) -> None:
    run = run_suite(core_suite, reference_agent())
    trace = run.by_case()["core-price-with-tax"][0].trace
    assert [call.tool for call in trace.tool_calls] == ["get_product", "calculator"]
    assert trace.tool_calls[1].result == {"value": 127.98}
    assert trace.usage.llm_calls >= 2 and trace.usage.input_tokens > 0
    assert trace.latency_ms > 0
    assert trace.status == "completed" and trace.status_source == "declared"


def test_same_seed_gives_the_same_fingerprint(core_suite: Suite) -> None:
    first = run_suite(core_suite, reference_agent(), RunConfig(seed=7, trials=2))
    second = run_suite(core_suite, reference_agent(), RunConfig(seed=7, trials=2))
    assert first.fingerprint() == second.fingerprint()
    assert len(first.results) == 2 * len(core_suite.cases)
