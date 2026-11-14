"""Model validation: the data layer must reject anything that would silently weaken a test."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_reliability_lab import reference_agent, run_suite
from agent_reliability_lab.models import (
    AgentTask,
    AnswerCheck,
    Attack,
    Budget,
    Case,
    CaseResult,
    Evaluation,
    Expected,
    FaultSpec,
    Finding,
    ModelPrice,
    ModelUsage,
    Mutation,
    PriceTable,
    Rule,
    RunConfig,
    SimPlan,
    StateAssertion,
    Suite,
    ToolCallSpec,
    Usage,
    WorkflowExpectation,
    default_price_table,
)
from helpers import case_data, make_case


@pytest.mark.parametrize(
    "model, data",
    [
        (Case, {**case_data(), "expeced": {}}),
        (Expected, {"forbiden": []}),
        (AnswerCheck, {"contains": ["x"]}),
        (Rule, {"max_word": 3}),
        (FaultSpec, {"kind": "timeout", "tol": "x"}),
        (Suite, {"name": "s", "cases": [case_data()], "default": {}}),
    ],
)
def test_unknown_fields_are_errors_not_ignored(model: type, data: dict) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(data)


@pytest.mark.parametrize("bad_id", ["", "../x", "a b", "a/b", ".hidden", "x" * 101, "a\nb"])
def test_case_ids_are_identifiers(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        make_case(id=bad_id)


def test_wrong_types_are_not_coerced_into_strings() -> None:
    with pytest.raises(ValidationError):
        make_case(input=12345)
    with pytest.raises(ValidationError):
        make_case(id=7)


def test_agent_task_has_no_way_to_carry_the_oracle() -> None:
    fields = set(AgentTask.model_fields)
    assert fields == {"input", "instructions", "context", "output_schema", "sim"}
    assert not {"expected", "attack", "canaries", "id"} & fields


@pytest.mark.parametrize(
    "data, message",
    [
        ({"kind": "dependency_failure"}, "needs a 'dependency'"),
        ({"kind": "timeout", "dependency": "payments"}, "only valid for dependency_failure"),
        ({"kind": "retry_exhaustion", "times": 2}, "affects every call"),
        ({"kind": "incorrect_response"}, "needs a mutation"),
        ({"kind": "incorrect_response", "mutation": {"op": "null"}}, "needs a mutation"),
        ({"kind": "invalid_output", "mutation": {"op": "negate", "field": "x"}}, "accepts"),
        ({"kind": "timeout", "mutation": {"op": "null"}}, "not valid"),
        ({"kind": "timeout", "probability": 1.5}, "less than or equal to 1"),
        ({"kind": "timeout", "times": 0}, "greater than or equal to 1"),
    ],
)
def test_inconsistent_fault_specs_are_rejected(data: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        FaultSpec.model_validate(data)


def test_mutations_need_their_parameters() -> None:
    with pytest.raises(ValidationError, match="needs a 'field'"):
        Mutation(op="negate")
    with pytest.raises(ValidationError, match="needs a 'factor'"):
        Mutation(op="scale", field="total")
    assert Mutation(op="scale", field="total", factor=2.0).factor == 2.0


def test_workflow_dependencies_must_exist_and_not_be_circular() -> None:
    with pytest.raises(ValidationError, match="unknown step"):
        WorkflowExpectation.model_validate(
            {"steps": [{"id": "a", "tool": "get_order", "after": ["missing"]}]}
        )
    with pytest.raises(ValidationError, match="depends on itself"):
        WorkflowExpectation.model_validate(
            {"steps": [{"id": "a", "tool": "get_order", "after": ["a"]}]}
        )
    with pytest.raises(ValidationError, match="must be unique"):
        WorkflowExpectation.model_validate(
            {"steps": [{"id": "a", "tool": "x"}, {"id": "a", "tool": "y"}]}
        )
    with pytest.raises(ValidationError, match="needs steps or final_state"):
        WorkflowExpectation.model_validate({})


def test_state_assertion_needs_exactly_one_check() -> None:
    assert StateAssertion(path=["emails"], length=0).length == 0
    assert StateAssertion(path=["x"], equals=None).equals is None
    with pytest.raises(ValidationError, match="exactly one"):
        StateAssertion(path=["emails"])
    with pytest.raises(ValidationError, match="exactly one"):
        StateAssertion(path=["emails"], length=1, contains="x")


def test_regular_expressions_are_checked_when_the_suite_is_loaded() -> None:
    with pytest.raises(ValidationError, match="invalid regular expression"):
        AnswerCheck(regex=["(unclosed"])
    with pytest.raises(ValidationError, match="longer than"):
        Rule(must_match="a" * 501)
    with pytest.raises(ValidationError, match="invalid regular expression"):
        Rule(must_not_match="[")


def test_canaries_must_be_long_enough_to_be_unambiguous() -> None:
    with pytest.raises(ValidationError, match="at least 8 characters"):
        make_case(canaries=["short"])


def test_attack_needs_an_indicator_and_call_counts_must_be_consistent() -> None:
    with pytest.raises(ValidationError, match="at least one success indicator"):
        Attack.model_validate({"kind": "direct", "goal": "x", "indicators": {}})
    with pytest.raises(ValidationError, match="max_count must be >= min_count"):
        ToolCallSpec(tool="get_order", min_count=2, max_count=1)
    with pytest.raises(ValidationError, match="at least one acceptable status"):
        Expected(status=[])


def test_sim_plan_validation() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        SimPlan.model_validate({"steps": [{"id": "a", "tool": "x"}, {"id": "a", "tool": "y"}]})
    with pytest.raises(ValidationError, match="clarifying_question"):
        SimPlan.model_validate({"missing": ["order_id"]})


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate case id"):
        Suite.model_validate({"name": "s", "cases": [case_data(), case_data()]})
    with pytest.raises(ValidationError, match="duplicate document id"):
        make_case(environment={"documents": [{"id": "d", "text": "a"}, {"id": "d", "text": "b"}]})
    with pytest.raises(ValidationError, match="instruction ids must be unique"):
        make_case(instructions=[{"id": "i", "text": "a"}, {"id": "i", "text": "b"}])


def test_substantive_checks_lists_what_can_fail() -> None:
    assert make_case().substantive_checks() == ["answer"]
    assert make_case(expected={}).substantive_checks() == []
    assert make_case(expected={"answer": {}}).substantive_checks() == []
    rich = make_case(
        output_schema={"type": "object"},
        instructions=[{"id": "i", "text": "short", "rule": {"max_words": 5}}],
        attack={"kind": "direct", "goal": "g", "indicators": {"output_contains": ["x"]}},
        expected={
            "status": "refused",
            "tool_calls": {"forbidden": [{"tool": "send_email"}]},
            "groundedness": {},
            "workflow": {"final_state": [{"path": ["emails"], "length": 0}]},
            "recovery": {"outcome": "failed_safely"},
        },
    )
    assert rich.substantive_checks() == [
        "status",
        "tool_calls",
        "groundedness",
        "workflow",
        "recovery",
        "output_schema",
        "attack",
        "instructions",
    ]


def test_budget_merge_prefers_the_case_value() -> None:
    base = Budget(max_latency_ms=15000, max_cost_usd=0.05)
    merged = Budget(max_cost_usd=0.2).merged_over(base)
    assert merged.max_cost_usd == 0.2 and merged.max_latency_ms == 15000
    assert Budget(max_tokens=10).merged_over(None).max_tokens == 10


def test_price_table_estimates_and_reports_missing_prices() -> None:
    usage = Usage(
        input_tokens=2_000_000,
        output_tokens=1_000_000,
        llm_calls=2,
        by_model={"sim-1": ModelUsage(input_tokens=2_000_000, output_tokens=1_000_000, calls=2)},
    )
    assert default_price_table().estimate(usage) == pytest.approx(2 * 1.0 + 1 * 5.0)
    assert PriceTable().estimate(usage) is None
    fallback = PriceTable(default=ModelPrice(input_per_mtok=3.0, output_per_mtok=0.0))
    assert fallback.estimate(usage) == pytest.approx(6.0)
    assert PriceTable().estimate(Usage()) == 0.0


def test_run_config_limits() -> None:
    with pytest.raises(ValidationError):
        RunConfig(trials=0)
    with pytest.raises(ValidationError):
        RunConfig(timeout_s=0)
    with pytest.raises(ValidationError):
        RunConfig(fault_profile="../../etc")


def test_fingerprint_ignores_wall_clock_run_id_and_creation_time(core_suite: Suite) -> None:
    first = run_suite(core_suite, reference_agent(), run_id="one")
    second = run_suite(core_suite, reference_agent(), run_id="two")
    assert first.fingerprint() == second.fingerprint()
    changed = second.model_copy(deep=True)
    changed.results[0].trace.output += "!"
    assert changed.fingerprint() != first.fingerprint()
    slower = second.model_copy(deep=True)
    slower.results[0].trace.wall_ms += 1000
    slower.created_at = "2000-01-01T00:00:00Z"
    assert slower.fingerprint() == first.fingerprint()


def test_stored_identifiers_are_validated_on_load(core_suite: Suite) -> None:
    run = run_suite(core_suite, reference_agent(), run_id="ok")
    data = run.model_dump(mode="json")
    for mutate in (
        lambda d: d.__setitem__("created_at", "yesterday <b>"),
        lambda d: d.__setitem__("suite_name", "../x"),
        lambda d: d["results"][0].__setitem__("category", "![x](http://a.example)"),
        lambda d: d["results"][0]["evaluations"][0].__setitem__("evaluator", "Bad Name"),
        lambda d: d["results"][0].__setitem__("failure_categories", ["<script>"]),
    ):
        broken = run.model_dump(mode="json")
        mutate(broken)
        with pytest.raises(ValidationError):
            type(run).model_validate(broken)
    assert type(run).model_validate(data).run_id == "ok"


def test_helpers_on_results() -> None:
    evaluation = Evaluation(
        evaluator="task_success",
        passed=False,
        findings=[
            Finding(code="wrong_status", message="x"),
            Finding(code="note", message="y", severity="info"),
        ],
    )
    assert [item.code for item in evaluation.failures()] == ["wrong_status"]
    assert CaseResult.model_fields["trace"].is_required()
