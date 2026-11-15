"""Fault injection: every kind behaves as documented, and everything is reproducible."""

from __future__ import annotations

import json

import pytest

from agent_reliability_lab.errors import ToolError
from agent_reliability_lab.faults import (
    FAULT_PROFILES,
    FaultInjector,
    apply_mutation,
    derive_seed,
    uniform,
)
from agent_reliability_lab.models import Environment, FaultSpec, Mutation
from agent_reliability_lab.sandbox import Sandbox
from agent_reliability_lab.tools import BUILTIN_TOOLS, World, resolve_tools
from helpers import ORDERS

TOOLS = [
    "get_order",
    "issue_refund",
    "send_email",
    "send_bulk_email",
    "search_docs",
    "list_orders",
    "calculator",
]


def sandbox(*faults: dict, seed: int = 0) -> Sandbox:
    environment = Environment.model_validate({"records": {"orders": ORDERS}})
    specs = [FaultSpec.model_validate(fault) for fault in faults]
    return Sandbox(
        World.from_environment(environment), resolve_tools(TOOLS), FaultInjector(specs, seed)
    )


def outcome(box: Sandbox, tool: str = "get_order", **arguments: object) -> str:
    try:
        box.call(tool, arguments or {"order_id": "A-1001"})
    except ToolError as error:
        return error.code
    return "ok"


def test_seeds_are_stable_and_spread() -> None:
    assert derive_seed(0, "case", 1) == derive_seed(0, "case", 1)
    assert derive_seed(0, "case", 1) != derive_seed(0, "case", 2)
    assert 0 <= derive_seed("x") < 2**63
    draws = [uniform("seed", n) for n in range(2000)]
    assert all(0.0 <= value < 1.0 for value in draws)
    assert 0.45 < sum(draws) / len(draws) < 0.55


def test_timeout_takes_simulated_time_then_fails_retryably() -> None:
    box = sandbox({"kind": "timeout", "tool": "get_order", "times": 1, "delay_ms": 4000})
    with pytest.raises(ToolError) as caught:
        box.call("get_order", {"order_id": "A-1001"})
    assert (caught.value.code, caught.value.retryable) == ("timeout", True)
    assert box.now_ms == 4000 and box.tool_calls[0].fault == "timeout"
    assert outcome(box) == "ok"


def test_default_timeout_is_ten_simulated_seconds() -> None:
    box = sandbox({"kind": "timeout"})
    assert outcome(box) == "timeout" and box.now_ms == 10_000


def test_transient_error_fails_the_first_n_calls() -> None:
    box = sandbox({"kind": "transient_error", "tool": "get_order", "times": 2})
    assert [outcome(box) for _ in range(4)] == ["unavailable", "unavailable", "ok", "ok"]
    assert [call.is_retry for call in box.tool_calls] == [False, True, True, False]


def test_after_skips_the_first_calls() -> None:
    box = sandbox({"kind": "transient_error", "tool": "get_order", "after": 2, "times": 1})
    assert [outcome(box) for _ in range(4)] == ["ok", "ok", "unavailable", "ok"]


def test_permanent_error_is_not_retryable_and_never_recovers() -> None:
    box = sandbox({"kind": "permanent_error", "tool": "get_order"})
    with pytest.raises(ToolError) as caught:
        box.call("get_order", {"order_id": "A-1001"})
    assert (caught.value.code, caught.value.retryable) == ("internal_error", False)
    assert [outcome(box) for _ in range(3)] == ["internal_error"] * 3


def test_retry_exhaustion_is_retryable_but_never_recovers() -> None:
    box = sandbox({"kind": "retry_exhaustion", "tool": "get_order"})
    with pytest.raises(ToolError) as caught:
        box.call("get_order", {"order_id": "A-1001"})
    assert (caught.value.code, caught.value.retryable) == ("unavailable", True)
    assert [outcome(box) for _ in range(5)] == ["unavailable"] * 5


def test_rate_limit_requires_waiting() -> None:
    box = sandbox({"kind": "rate_limit", "tool": "get_order", "times": 1, "retry_after_ms": 1000})
    with pytest.raises(ToolError) as caught:
        box.call("get_order", {"order_id": "A-1001"})
    assert caught.value.code == "rate_limited" and caught.value.retry_after_ms == 1000
    assert outcome(box) == "rate_limited", "calling again too early is rejected again"
    assert outcome(box) == "rate_limited"
    box.wait(1000)
    assert outcome(box) == "ok"


def test_rate_limit_does_not_affect_other_tools() -> None:
    box = sandbox({"kind": "rate_limit", "tool": "get_order", "times": 1})
    assert outcome(box) == "rate_limited"
    assert outcome(box, "calculator", expression="1+1") == "ok"


def test_dependency_failure_takes_down_every_dependent_tool() -> None:
    box = sandbox({"kind": "dependency_failure", "dependency": "orders-db"})
    assert BUILTIN_TOOLS["issue_refund"].depends_on == ("payments", "orders-db")
    assert outcome(box, "get_order", order_id="A-1001") == "dependency_failed"
    assert outcome(box, "list_orders", customer_id="C-100") == "dependency_failed"
    assert outcome(box, "issue_refund", order_id="A-1002", amount=1) == "dependency_failed"
    assert outcome(box, "calculator", expression="1+1") == "ok"
    assert outcome(box, "send_email", to="a@example.com", subject="s", body="b") == "ok"
    error = box.tool_calls[0].error
    assert error is not None and "orders-db" in error.message and error.retryable is False


def test_incorrect_response_is_silent() -> None:
    box = sandbox(
        {
            "kind": "incorrect_response",
            "tool": "get_order",
            "mutation": {"op": "negate", "field": "total"},
        }
    )
    result = box.call("get_order", {"order_id": "A-1002"})
    assert result["total"] == -129.5 and box.tool_calls[0].ok is True
    assert box.tool_calls[0].fault == "incorrect_response"
    assert box.world.records["orders"]["A-1002"]["total"] == 129.5, "the world itself is untouched"


def test_invalid_output_defaults_to_truncated_json() -> None:
    box = sandbox({"kind": "invalid_output", "tool": "get_order", "times": 1})
    broken = box.call("get_order", {"order_id": "A-1001"})
    assert isinstance(broken, str)
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    assert isinstance(box.call("get_order", {"order_id": "A-1001"}), dict)


def test_partial_response_only_applies_to_tools_that_can_complete_partially() -> None:
    box = sandbox({"kind": "partial_response", "times": 1})
    assert box.call("get_order", {"order_id": "A-1001"})["status"] == "shipped"
    assert box.tool_calls[0].fault is None
    recipients = [f"u{n}@example.com" for n in range(4)]
    result = box.call("send_bulk_email", {"recipients": recipients, "subject": "s", "body": "b"})
    assert result == {"sent": recipients[:2], "failed": recipients[2:]}
    assert len(box.world.emails) == 2 and box.tool_calls[1].fault == "partial_response"
    again = box.call("send_bulk_email", {"recipients": recipients[2:], "subject": "s", "body": "b"})
    assert again["failed"] == [] and len(box.world.emails) == 4


def test_first_matching_fault_wins_and_tool_filter_applies() -> None:
    box = sandbox(
        {"kind": "permanent_error", "tool": "issue_refund"},
        {"kind": "transient_error", "tool": "*", "times": 1},
    )
    assert outcome(box, "issue_refund", order_id="A-1002", amount=1) == "internal_error"
    assert outcome(box) == "unavailable"
    assert outcome(box) == "ok"


def test_probabilistic_faults_are_reproducible_and_seed_dependent() -> None:
    def pattern(seed: int) -> list[str]:
        box = sandbox({"kind": "transient_error", "probability": 0.5}, seed=seed)
        return [outcome(box) for _ in range(40)]

    assert pattern(1) == pattern(1)
    assert pattern(1) != pattern(2)
    failures = pattern(1).count("unavailable")
    assert 10 <= failures <= 30


def test_fault_decision_does_not_depend_on_other_tools_being_called() -> None:
    spec = {"kind": "transient_error", "tool": "get_order", "probability": 0.5}
    alone, mixed = sandbox(spec, seed=7), sandbox(spec, seed=7)
    first = [outcome(alone) for _ in range(20)]
    second = []
    for _ in range(20):
        outcome(mixed, "calculator", expression="1+1")
        second.append(outcome(mixed))
    assert first == second


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"op": "negate", "field": "total"}, {"id": "A", "total": -10, "nested": {"n": 2.5}}),
        (
            {"op": "scale", "field": "total", "factor": 100},
            {"id": "A", "total": 1000, "nested": {"n": 2.5}},
        ),
        (
            {"op": "scale", "field": "nested.n", "factor": 2},
            {"id": "A", "total": 10, "nested": {"n": 5.0}},
        ),
        (
            {"op": "set", "field": "id", "value": None},
            {"id": None, "total": 10, "nested": {"n": 2.5}},
        ),
        ({"op": "drop_field", "field": "total"}, {"id": "A", "nested": {"n": 2.5}}),
        ({"op": "negate", "field": "missing"}, {"id": "A", "total": 10, "nested": {"n": 2.5}}),
        ({"op": "negate", "field": "id"}, {"id": "A", "total": 10, "nested": {"n": 2.5}}),
        ({"op": "null"}, None),
        ({"op": "wrong_type"}, [["id", "A"], ["total", 10], ["nested", {"n": 2.5}]]),
    ],
)
def test_mutations(mutation: dict, expected: object) -> None:
    original = {"id": "A", "total": 10, "nested": {"n": 2.5}}
    assert apply_mutation(original, Mutation.model_validate(mutation)) == expected
    assert original == {"id": "A", "total": 10, "nested": {"n": 2.5}}


def test_stringify_and_truncate_mutations() -> None:
    value = {"id": "A-1001", "status": "shipped"}
    text = apply_mutation(value, Mutation(op="stringify"))
    assert json.loads(text) == value
    cut = apply_mutation(value, Mutation(op="truncate_json"))
    assert isinstance(cut, str) and text.startswith(cut) and len(cut) < len(text)
    assert apply_mutation(5, Mutation(op="wrong_type")) == "5"


def test_profiles_are_valid_probabilistic_faults() -> None:
    assert sorted(FAULT_PROFILES) == ["degraded", "flaky-tools", "rate-limited", "slow-network"]
    for specs in FAULT_PROFILES.values():
        assert specs and all(0 < spec.probability < 1 and spec.tool == "*" for spec in specs)
