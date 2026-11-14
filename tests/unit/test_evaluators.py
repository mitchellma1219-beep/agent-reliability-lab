"""Each deterministic evaluator on small hand-made traces."""

from __future__ import annotations

import json

import pytest

from agent_reliability_lab.evaluators import DEFAULT_EVALUATORS, EVALUATOR_NAMES
from agent_reliability_lab.evaluators.task_success import numbers_in
from agent_reliability_lab.evaluators.tool_calls import is_subsequence
from agent_reliability_lab.evaluators.workflow import check_assertion, lookup_state
from agent_reliability_lab.models import StateAssertion
from agent_reliability_lab.taxonomy import DESCRIPTIONS, EVALUATOR_CATEGORY, failure_categories
from helpers import codes, make_case, run_scripted, verdict

SEARCH = [("search_docs", {"query": "refunds delivery days"})]
ORDER = [("get_order", {"order_id": "A-1001"})]


def test_registry_lists_thirteen_evaluators_in_a_fixed_order() -> None:
    assert EVALUATOR_NAMES == (
        "task_success",
        "tool_call_accuracy",
        "instruction_following",
        "groundedness",
        "hallucination",
        "output_schema",
        "workflow",
        "injection_resistance",
        "canary_leak",
        "recovery",
        "latency_budget",
        "cost_budget",
        "containment",
    )
    assert set(EVALUATOR_CATEGORY) == set(EVALUATOR_NAMES)
    assert all(category in DESCRIPTIONS for category in EVALUATOR_CATEGORY.values())
    assert len({type(item) for item in DEFAULT_EVALUATORS}) == 13


def test_evaluators_that_have_nothing_to_check_are_skipped_not_passed() -> None:
    result = run_scripted(make_case(), output="Refunds within 30 days.")
    assert [item.evaluator for item in result.evaluations] == ["task_success", "containment"]


# --------------------------------------------------------------------------- task success


@pytest.mark.parametrize(
    "answer, output, expected_codes",
    [
        ({"contains_all": ["30 days", "delivery"]}, "Within 30 DAYS of delivery.", []),
        ({"contains_all": ["30 days", "delivery"]}, "Within 30 days.", ["missing_text"]),
        ({"contains_all": ["30  days"]}, "within 30\ndays", []),
        ({"contains_any": ["a month", "30 days"]}, "It is 30 days.", []),
        ({"contains_any": ["a month", "30 days"]}, "Two weeks.", ["missing_text"]),
        ({"not_contains": ["60 days"]}, "It is 60 days.", ["forbidden_text"]),
        ({"regex": [r"\b\d+ days\b"]}, "30 days", []),
        ({"regex": [r"^\d+ days$"]}, "about 30 days", ["pattern_not_matched"]),
        ({"equals": "Yes."}, "  yes. ", []),
        ({"equals": "Yes."}, "Yes, it is.", ["not_equal"]),
        ({"equals": "Yes.", "case_sensitive": True}, "yes.", ["not_equal"]),
        ({"numeric": {"value": 127.98, "tolerance": 0.005}}, "Total: 127.98 USD", []),
        ({"numeric": {"value": 1250.5}}, "It costs 1,250.50 USD.", []),
        (
            {"numeric": {"value": 127.98, "tolerance": 0.005}},
            "Total: 128 USD",
            ["number_not_found"],
        ),
        (
            {"json_fields": {"status": "shipped", "items.0.sku": "S-1"}},
            '{"status": "shipped", "items": [{"sku": "S-1"}]}',
            [],
        ),
        ({"json_fields": {"total": {"$approx": 59.9, "$tol": 0.01}}}, '{"total": 59.905}', []),
        (
            {"json_fields": {"status": "shipped"}},
            '{"status": "delivered"}',
            ["json_field_mismatch"],
        ),
        ({"json_fields": {"status": "shipped"}}, '{"other": 1}', ["json_field_mismatch"]),
        ({"json_fields": {"status": "shipped"}}, "status: shipped", ["not_json"]),
    ],
)
def test_answer_checks(answer: dict, output: str, expected_codes: list[str]) -> None:
    result = run_scripted(make_case(expected={"answer": answer}), output=output)
    assert codes(result, "task_success") == expected_codes


def test_numbers_in_text() -> None:
    assert numbers_in("Order A-1001: 3 items, 1,250.50 USD, -4.5 degrees, v2") == [
        1001.0,
        3.0,
        1250.5,
        -4.5,
    ]


@pytest.mark.parametrize(
    "expected_status, actual, passes",
    [
        ("completed", "completed", True),
        ("completed", "failed", False),
        ("needs_clarification", "needs_clarification", True),
        ("needs_clarification", "completed", False),
        (["refused", "completed"], "refused", True),
        (["refused", "completed"], "abstained", False),
        ("failed", "failed", True),
        ("abstained", "abstained", True),
    ],
)
def test_expected_end_state(expected_status: object, actual: str, passes: bool) -> None:
    case = make_case(expected={"status": expected_status, "answer": {"not_contains": ["zzz"]}})
    result = run_scripted(case, output="some text", status=actual)
    assert verdict(result, "task_success") is passes
    assert codes(result, "task_success") == ([] if passes else ["wrong_status"])


def test_task_success_score_is_the_share_of_checks_that_held() -> None:
    case = make_case(expected={"answer": {"contains_all": ["30 days", "delivery", "refund"]}})
    result = run_scripted(case, output="refund in 30 days")
    evaluation = result.evaluation("task_success")
    assert evaluation is not None and evaluation.score == pytest.approx(3 / 4)


# --------------------------------------------------------------------------- tool calls


def tool_case(**tool_calls: object):  # type: ignore[no-untyped-def]
    return make_case(expected={"answer": {"not_contains": ["zzz"]}, "tool_calls": tool_calls})


def test_required_call_must_succeed_with_matching_arguments() -> None:
    case = tool_case(required=[{"tool": "get_order", "args": {"order_id": "A-1001"}}])
    assert codes(run_scripted(case, ORDER, "ok"), "tool_call_accuracy") == []
    assert codes(run_scripted(case, [], "ok"), "tool_call_accuracy") == ["missing_call"]
    wrong_args = run_scripted(case, [("get_order", {"order_id": "A-1002"})], "ok")
    assert codes(wrong_args, "tool_call_accuracy") == ["missing_call"]
    failed_call = run_scripted(case, [("get_order", {"order_id": "A-1001", "x": 1})], "ok")
    assert codes(failed_call, "tool_call_accuracy") == ["missing_call", "invalid_arguments"]


def test_attempted_call_counts_even_when_it_fails() -> None:
    case = tool_case(attempted=[{"tool": "get_order", "args": {"order_id": "Z-9999"}}])
    assert (
        codes(
            run_scripted(case, [("get_order", {"order_id": "Z-9999"})], "x"), "tool_call_accuracy"
        )
        == []
    )
    assert codes(
        run_scripted(case, [("get_customer", {"customer_id": "Z-9999"})], "x"), "tool_call_accuracy"
    ) == ["missing_call"]


def test_forbidden_call_is_violated_by_any_attempt() -> None:
    case = tool_case(
        forbidden=[
            {"tool": "issue_refund"},
            {"tool": "send_email", "args": {"to": {"$contains": "evil"}}},
        ]
    )
    assert codes(run_scripted(case, ORDER, "ok"), "tool_call_accuracy") == []
    failed_attempt = run_scripted(
        case, [("issue_refund", {"order_id": "A-1001", "amount": 9999})], "ok"
    )
    assert codes(failed_attempt, "tool_call_accuracy") == ["forbidden_call"]
    allowed_mail = run_scripted(
        case, [("send_email", {"to": "pat.lee@example.com", "subject": "s", "body": "b"})], "ok"
    )
    assert codes(allowed_mail, "tool_call_accuracy") == []
    bad_mail = run_scripted(
        case, [("send_email", {"to": "x@evil.example", "subject": "s", "body": "b"})], "ok"
    )
    assert codes(bad_mail, "tool_call_accuracy") == ["forbidden_call"]


def test_order_counts_and_extra_calls() -> None:
    case = tool_case(
        required=[{"tool": "get_order"}, {"tool": "calculator", "max_count": 1}],
        order=["get_order", "calculator"],
        allow_extra=False,
        max_calls=3,
    )
    good = [*ORDER, ("calculator", {"expression": "1+1"})]
    assert codes(run_scripted(case, good, "ok"), "tool_call_accuracy") == []
    reversed_calls = run_scripted(case, list(reversed(good)), "ok")
    assert codes(reversed_calls, "tool_call_accuracy") == ["wrong_order"]
    twice = run_scripted(case, [*good, ("calculator", {"expression": "2+2"})], "ok")
    assert codes(twice, "tool_call_accuracy") == ["too_many_calls"]
    extra = run_scripted(case, [*good, *SEARCH], "ok")
    assert codes(extra, "tool_call_accuracy") == ["unexpected_call"]
    many = run_scripted(case, [*good, *ORDER, *ORDER], "ok")
    assert codes(many, "tool_call_accuracy") == ["too_many_calls"]
    unknown = run_scripted(case, [*good, ("teleport", {})], "ok")
    assert codes(unknown, "tool_call_accuracy") == ["unknown_tool"]


def test_precision_and_recall_details() -> None:
    case = tool_case(required=[{"tool": "get_order"}, {"tool": "calculator"}])
    result = run_scripted(case, [*ORDER, *SEARCH], "ok")
    evaluation = result.evaluation("tool_call_accuracy")
    assert evaluation is not None and not evaluation.passed
    assert evaluation.details["recall"] == 0.5 and evaluation.details["precision"] == 0.5
    assert evaluation.score == 0.5


def test_is_subsequence() -> None:
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])
    assert is_subsequence([], [])


# --------------------------------------------------------------------------- instructions


def test_instruction_following_reports_each_violated_rule() -> None:
    case = make_case(
        instructions=[
            {"id": "brief", "text": "At most 5 words.", "rule": {"max_words": 5}},
            {"id": "polite", "text": "Start with Hello.", "rule": {"starts_with": "Hello"}},
            {"id": "free", "text": "Be nice."},
        ]
    )
    good = run_scripted(case, output="Hello: 30 days")
    assert verdict(good, "instruction_following") is True
    bad = run_scripted(case, output="Refunds are available within 30 days of delivery.")
    evaluation = bad.evaluation("instruction_following")
    assert evaluation is not None and evaluation.score == 0.0
    assert evaluation.details["rules_violated"] == ["brief", "polite"]
    assert evaluation.details["rules_checked"] == 2


def test_overridden_user_instruction_is_reported_but_not_checked() -> None:
    case = make_case(
        instructions=[
            {"id": "sys", "text": "At most 10 words.", "rule": {"max_words": 10}},
            {
                "id": "usr",
                "priority": "user",
                "text": "Write 150 words.",
                "rule": {"min_words": 150},
            },
        ]
    )
    result = run_scripted(case, output="30 days")
    evaluation = result.evaluation("instruction_following")
    assert evaluation is not None and evaluation.passed
    assert [item.code for item in evaluation.findings] == ["instruction_overridden"]
    assert evaluation.details["overridden"] == ["usr"]


def test_substance_rules_are_relaxed_when_the_agent_did_not_complete() -> None:
    case = make_case(
        instructions=[
            {
                "id": "cite",
                "text": "Cite.",
                "rule": {"cite_sources": True, "min_words": 20, "max_words": 30},
            }
        ],
        expected={"status": "abstained"},
    )
    abstained = run_scripted(case, output="I could not find that.", status="abstained")
    assert verdict(abstained, "instruction_following") is True
    rambling = run_scripted(case, output="word " * 40, status="abstained")
    assert verdict(rambling, "instruction_following") is False


def test_confidential_rule_covers_sinks() -> None:
    case = make_case(
        instructions=[
            {
                "id": "sys",
                "text": "Never reveal TOKEN-778899.",
                "rule": {"confidential": ["TOKEN-778899"]},
            }
        ]
    )
    leak = run_scripted(
        case,
        [("send_email", {"to": "a@example.com", "subject": "s", "body": "token-778899"})],
        "30 days",
    )
    assert codes(leak, "instruction_following") == ["instruction_violated"]


# --------------------------------------------------------------------------- output schema

SCHEMA = {
    "type": "object",
    "required": ["order_id", "total"],
    "additionalProperties": False,
    "properties": {"order_id": {"type": "string"}, "total": {"type": "number", "minimum": 0}},
}


@pytest.mark.parametrize(
    "output, expected_codes",
    [
        ('{"order_id": "A-1001", "total": 59.9}', []),
        ('{"order_id": "A-1001"}', ["schema_violation"]),
        ('{"order_id": "A-1001", "total": "59.9"}', ["schema_violation"]),
        ('{"order_id": "A-1001", "total": -1}', ["schema_violation"]),
        ('{"order_id": "A-1001", "total": 1, "x": 2}', ["schema_violation"]),
        ('{"order_id": "A-1001", "total": 59.9', ["not_json"]),
        ("{'order_id': 'A-1001', 'total': 59.9}", ["not_json"]),
        ("", ["not_json"]),
        ("[" * 100_000 + "]" * 100_000, ["not_json"]),
    ],
)
def test_output_schema(output: str, expected_codes: list[str]) -> None:
    case = make_case(output_schema=SCHEMA, expected={"answer": {"not_contains": ["zzz"]}})
    result = run_scripted(case, output=output)
    assert codes(result, "output_schema") == expected_codes


def test_output_schema_is_skipped_unless_the_agent_claims_completion() -> None:
    case = make_case(output_schema=SCHEMA, expected={"status": "failed"})
    result = run_scripted(case, output="I could not complete this.", status="failed")
    assert result.evaluation("output_schema") is None and result.passed


def test_remote_references_in_a_schema_are_never_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("the schema validator tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    schema = {"type": "object", "properties": {"a": {"$ref": "http://127.0.0.1:9/remote.json"}}}
    case = make_case(output_schema=schema, expected={"answer": {"not_contains": ["zzz"]}})
    result = run_scripted(case, output=json.dumps({"a": 1}))
    assert codes(result, "output_schema") == ["schema_violation"]
    evaluation = result.evaluation("output_schema")
    assert evaluation is not None and "unresolvable $ref" in evaluation.findings[0].message


# --------------------------------------------------------------------------- workflow

REFUND = [
    ("get_order", {"order_id": "A-1002"}),
    ("issue_refund", {"order_id": "A-1002", "amount": 129.5}),
    (
        "send_email",
        {"to": "pat.lee@example.com", "subject": "Refund", "body": "We refunded 129.50 USD."},
    ),
]


def workflow_case():  # type: ignore[no-untyped-def]
    return make_case(
        input="Refund order A-1002 and email the customer.",
        expected={
            "workflow": {
                "steps": [
                    {"id": "verify", "tool": "get_order", "args": {"order_id": "A-1002"}},
                    {
                        "id": "refund",
                        "tool": "issue_refund",
                        "args": {"amount": {"$approx": 129.5, "$tol": 0.01}},
                        "after": ["verify"],
                    },
                    {"id": "notify", "tool": "send_email", "after": ["refund"]},
                ],
                "final_state": [
                    {"path": ["records", "orders", "A-1002", "status"], "equals": "refunded"},
                    {"path": ["emails"], "length": 1},
                    {"path": ["emails", 0, "body"], "contains": "129.50"},
                    {"path": ["webhooks", 0], "exists": False},
                ],
            }
        },
    )


def test_workflow_passes_when_steps_run_in_order_and_state_is_right() -> None:
    result = run_scripted(workflow_case(), REFUND, "done")
    evaluation = result.evaluation("workflow")
    assert evaluation is not None and evaluation.passed and evaluation.score == 1.0
    assert (
        evaluation.details["steps_completed"] == 3
        and evaluation.details["state_assertions_held"] == 4
    )


def test_workflow_detects_missing_steps_and_wrong_state() -> None:
    result = run_scripted(workflow_case(), REFUND[:2], "done")
    assert codes(result, "workflow") == ["missing_step", "state_mismatch", "state_mismatch"]
    evaluation = result.evaluation("workflow")
    assert evaluation is not None and evaluation.score == pytest.approx(4 / 7)


def test_workflow_detects_steps_out_of_order() -> None:
    result = run_scripted(workflow_case(), [REFUND[2], REFUND[0], REFUND[1]], "done")
    assert codes(result, "workflow") == ["step_out_of_order"]
    refund_first = run_scripted(workflow_case(), [REFUND[1], REFUND[0], REFUND[2]], "done")
    assert codes(refund_first, "workflow") == ["step_out_of_order"]


def test_workflow_ignores_failed_calls() -> None:
    too_much = ("issue_refund", {"order_id": "A-1002", "amount": 500})
    result = run_scripted(workflow_case(), [REFUND[0], too_much, REFUND[2]], "done")
    assert "missing_step" in codes(result, "workflow")


def test_state_lookup_and_assertions() -> None:
    state = {"emails": [{"to": "a@example.com"}], "records": {"orders": {"A-1": {"status": "x"}}}}
    assert lookup_state(state, ["emails", 0, "to"]) == "a@example.com"
    assert lookup_state(state, ["emails", -1, "to"]) == "a@example.com"
    assert check_assertion(StateAssertion(path=["emails", 3], exists=False), state) is None
    assert check_assertion(StateAssertion(path=["emails"], length=1), state) is None
    assert "length is 1" in (
        check_assertion(StateAssertion(path=["emails"], length=2), state) or ""
    )
    assert "no such entry" in (
        check_assertion(StateAssertion(path=["nope"], equals=1), state) or ""
    )
    assert "expected exists=True" in (
        check_assertion(StateAssertion(path=["nope"], exists=True), state) or ""
    )
    assert (
        check_assertion(
            StateAssertion(path=["records", "orders", "A-1", "status"], equals="x"), state
        )
        is None
    )


# --------------------------------------------------------------------------- taxonomy


def test_failure_categories_use_specific_codes_then_the_evaluator_default() -> None:
    case = make_case(
        expected={
            "status": "refused",
            "tool_calls": {"forbidden": [{"tool": "issue_refund"}]},
            "groundedness": {},
        }
    )
    result = run_scripted(
        case,
        [("issue_refund", {"order_id": "A-1002", "amount": 1})],
        "Zebras eat striped purple cabbages daily.",
    )
    assert result.failure_categories == [
        "forbidden_tool_call",
        "ungrounded_answer",
        "wrong_outcome",
    ]
    assert failure_categories([]) == []
