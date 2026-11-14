"""Recovery from injected faults, and latency and cost budgets."""

from __future__ import annotations

from typing import Any

import pytest

from agent_reliability_lab import AgentContext, AgentResponse, FunctionAgent, ToolError, run_case
from agent_reliability_lab.models import CaseResult, ModelPrice, PriceTable, RunConfig
from helpers import codes, make_case, run_scripted, verdict

ORDER = ("get_order", {"order_id": "A-1001"})


def fault_case(fault: dict[str, Any], **recovery: Any):  # type: ignore[no-untyped-def]
    return make_case(
        input="What is the status of order A-1001?",
        faults=[{"tool": "get_order", **fault}],
        expected={"status": ["completed", "failed"], "recovery": recovery},
    )


def retrying_agent(attempts: int, wait_ms: float | None = 0.0, status_on_failure: str = "failed"):
    """An agent that tries the lookup up to ``attempts`` times, waiting between tries."""

    def run(ctx: AgentContext) -> AgentResponse:
        for attempt in range(attempts):
            try:
                order = ctx.tools.call("get_order", {"order_id": "A-1001"})
                return AgentResponse(
                    output=f"Order A-1001 is {order['status']}.", status="completed"
                )
            except ToolError as error:
                if attempt + 1 < attempts and wait_ms is not None:
                    ctx.wait(
                        error.retry_after_ms if wait_ms == -1 and error.retry_after_ms else wait_ms
                    )
        return AgentResponse(output="I could not look that up.", status=status_on_failure)  # type: ignore[arg-type]

    return FunctionAgent(run)


def run(case: Any, agent: FunctionAgent) -> CaseResult:
    return run_case(case, agent)


def test_retry_after_a_transient_fault_recovers() -> None:
    case = fault_case({"kind": "transient_error", "times": 1}, outcome="recovered", min_attempts=2)
    result = run(case, retrying_agent(3))
    evaluation = result.evaluation("recovery")
    assert evaluation is not None and evaluation.passed and evaluation.score == 1.0
    assert evaluation.details["faulted_calls"] == 1 and evaluation.details["retries"] == 1


def test_giving_up_on_a_retryable_fault_is_flagged() -> None:
    case = fault_case({"kind": "transient_error", "times": 1}, outcome="recovered", min_attempts=2)
    result = run(case, retrying_agent(1))
    assert codes(result, "recovery") == ["no_retry", "did_not_recover"]


def test_bounded_attempts() -> None:
    case = fault_case(
        {"kind": "retry_exhaustion"}, outcome="failed_safely", min_attempts=2, max_attempts=4
    )
    assert verdict(run(case, retrying_agent(3)), "recovery") is True
    assert verdict(run(case, retrying_agent(4)), "recovery") is True
    assert codes(run(case, retrying_agent(5)), "recovery") == ["too_many_attempts"]
    assert codes(run(case, retrying_agent(1)), "recovery") == ["no_retry"]


def test_runaway_retry_loop_is_stopped_and_flagged() -> None:
    case = fault_case({"kind": "retry_exhaustion"}, outcome="failed_safely", max_attempts=4)
    result = run_case(case, retrying_agent(10_000), RunConfig(max_tool_calls=20))
    assert codes(result, "recovery") == ["too_many_attempts", "runaway_retries"]
    assert result.trace.status == "failed" and result.trace.status_source == "harness"
    assert [error.kind for error in result.trace.errors] == ["harness_abort"]
    assert len(result.trace.tool_calls) == 20


def test_retrying_a_non_retryable_fault_is_flagged() -> None:
    case = fault_case({"kind": "permanent_error"}, outcome="failed_safely", max_attempts=1)
    assert verdict(run(case, retrying_agent(1)), "recovery") is True
    again = run(case, retrying_agent(2))
    assert codes(again, "recovery") == ["too_many_attempts", "retried_non_retryable"]


def test_fabricating_an_answer_after_a_fault_is_not_an_explicit_failure() -> None:
    case = fault_case({"kind": "permanent_error"}, outcome="failed_safely", max_attempts=1)
    result = run(case, retrying_agent(1, status_on_failure="completed"))
    assert codes(result, "recovery") == ["not_explicit_failure"]


def test_retry_after_must_be_respected() -> None:
    fault = {"kind": "rate_limit", "times": 1, "retry_after_ms": 1500}
    case = fault_case(fault, outcome="recovered", min_attempts=2)
    patient = run(case, retrying_agent(3, wait_ms=-1))
    assert verdict(patient, "recovery") is True
    impatient = run(case, retrying_agent(3, wait_ms=0))
    assert "ignored_retry_after" in codes(impatient, "recovery")
    relaxed = fault_case(fault, outcome="recovered", respect_retry_after=False, max_attempts=10)
    slow_enough = run(relaxed, retrying_agent(10, wait_ms=600))
    assert verdict(slow_enough, "recovery") is True


def test_using_a_corrupted_result_without_a_clean_retry_is_flagged() -> None:
    fault = {
        "kind": "incorrect_response",
        "mutation": {"op": "negate", "field": "total"},
        "times": 1,
    }
    case = make_case(
        input="Total of order A-1001?",
        faults=[{"tool": "get_order", **fault}],
        expected={"recovery": {"outcome": "recovered", "min_attempts": 2}},
    )
    trusting = run_scripted(case, [ORDER], "The total is -59.90 USD.")
    assert codes(trusting, "recovery") == ["no_retry", "used_faulty_result"]
    careful = run_scripted(case, [ORDER, ORDER], "The total is 59.90 USD.")
    assert verdict(careful, "recovery") is True


def test_recovery_only_judges_injected_faults() -> None:
    case = make_case(
        input="Status of order Z-404?",
        expected={
            "status": "abstained",
            "recovery": {"outcome": "failed_safely", "max_attempts": 1},
        },
    )
    result = run_scripted(
        case, [("get_order", {"order_id": "Z-404"})], "No such order.", "abstained"
    )
    evaluation = result.evaluation("recovery")
    assert evaluation is not None and evaluation.passed and evaluation.details["faulted_calls"] == 0


# --------------------------------------------------------------------------- budgets


def budget_case(**budget: Any):  # type: ignore[no-untyped-def]
    return make_case(expected={"answer": {"not_contains": ["zzz"]}, "budget": budget})


def test_latency_budget_uses_accounted_time() -> None:
    case = budget_case(max_latency_ms=1000)
    fast = run_scripted(case, [ORDER], "ok", input_tokens=10, output_tokens=5, latency_ms=900)
    assert verdict(fast, "latency_budget") is True and fast.trace.latency_ms == 960
    slow = run_scripted(case, [ORDER], "ok", input_tokens=10, output_tokens=5, latency_ms=2000)
    assert codes(slow, "latency_budget") == ["latency_exceeded"]
    evaluation = slow.evaluation("latency_budget")
    assert evaluation is not None and evaluation.score == pytest.approx(1000 / 2060)
    assert slow.evaluation("cost_budget") is None


def test_cost_budget_uses_the_price_table_and_says_it_is_an_estimate() -> None:
    case = budget_case(max_cost_usd=0.01)
    prices = PriceTable(
        name="test", models={"scripted": ModelPrice(input_per_mtok=10.0, output_per_mtok=30.0)}
    )
    config = RunConfig(prices=prices)
    cheap = run_scripted(case, [], "ok", config=config, input_tokens=500, output_tokens=100)
    assert (
        cheap.estimated_cost_usd == pytest.approx(0.008) and verdict(cheap, "cost_budget") is True
    )
    dear = run_scripted(case, [], "ok", config=config, input_tokens=5000, output_tokens=100)
    assert codes(dear, "cost_budget") == ["cost_exceeded"]
    evaluation = dear.evaluation("cost_budget")
    assert evaluation is not None and "estimated cost" in evaluation.findings[0].message


def test_cost_budget_fails_closed_when_a_model_has_no_price() -> None:
    result = run_scripted(budget_case(max_cost_usd=1.0), [], "ok", input_tokens=10, output_tokens=1)
    assert result.estimated_cost_usd is None
    assert codes(result, "cost_budget") == ["cost_unknown"]


def test_token_and_tool_call_budgets() -> None:
    case = budget_case(max_tokens=100, max_tool_calls=1)
    within = run_scripted(case, [ORDER], "ok", input_tokens=60, output_tokens=40)
    assert verdict(within, "cost_budget") is True
    over = run_scripted(case, [ORDER, ORDER], "ok", input_tokens=90, output_tokens=40)
    assert codes(over, "cost_budget") == ["tokens_exceeded", "tool_calls_exceeded"]
