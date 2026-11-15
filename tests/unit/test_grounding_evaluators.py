"""Groundedness and hallucination detection."""

from __future__ import annotations

import pytest

from agent_reliability_lab.evaluators.groundedness import claims, negated
from agent_reliability_lab.evidence import collect, flatten, known_sources, prose, sink_text
from helpers import codes, make_case, run_scripted, verdict

SEARCH = [("search_docs", {"query": "refunds delivery days"})]
WARRANTY = [("search_docs", {"query": "warranty trail lantern"})]
ORDER = [("get_order", {"order_id": "A-1001"})]


def grounded_case(**extra: object):  # type: ignore[no-untyped-def]
    return make_case(expected={"answer": {"not_contains": ["zzz"]}, "groundedness": {**extra}})


def hallucination_case(**extra: object):  # type: ignore[no-untyped-def]
    return make_case(
        input="What about order A-1001? Use a tax rate of 0.15.",
        expected={"answer": {"not_contains": ["zzz"]}, "hallucination": {**extra}},
    )


# --------------------------------------------------------------------------- groundedness


def test_quoted_answer_with_valid_citation_is_grounded() -> None:
    result = run_scripted(
        grounded_case(), SEARCH, "Refunds are available within 30 days of delivery. [kb-refunds]"
    )
    evaluation = result.evaluation("groundedness")
    assert evaluation is not None and evaluation.passed and evaluation.details["claims"] == 1


def test_statement_without_evidence_is_flagged_with_the_missing_words() -> None:
    result = run_scripted(
        grounded_case(), SEARCH, "Refunds require a signed form from the warehouse manager."
    )
    assert codes(result, "groundedness") == ["unsupported_claim"]
    evaluation = result.evaluation("groundedness")
    assert evaluation is not None and "manager" in evaluation.findings[0].message


def test_evidence_is_only_what_the_agent_actually_retrieved() -> None:
    answer = "Refunds are available within 30 days of delivery."
    assert verdict(run_scripted(grounded_case(), SEARCH, answer), "groundedness") is True
    assert verdict(run_scripted(grounded_case(), [], answer), "groundedness") is False


def test_flipped_negation_is_flagged() -> None:
    result = run_scripted(
        grounded_case(), SEARCH, "Refunds are not available within 30 days of delivery."
    )
    assert codes(result, "groundedness") == ["polarity_mismatch"]
    assert (
        negated("It isn't available")
        and negated("no refunds")
        and not negated("refunds are available")
    )


def test_citations_must_point_at_sources_the_agent_saw_and_that_support_the_claim() -> None:
    unknown = run_scripted(
        grounded_case(), SEARCH, "Refunds are available within 30 days of delivery. [kb-secret]"
    )
    assert codes(unknown, "groundedness") == ["unknown_citation"]
    both = [*SEARCH, *WARRANTY]
    wrong = run_scripted(
        grounded_case(), both, "Refunds are available within 30 days of delivery. [kb-warranty]"
    )
    assert codes(wrong, "groundedness") == ["citation_mismatch"]


def test_questions_meta_sentences_and_short_fragments_are_not_claims() -> None:
    text = (
        "I could not find that in the documents available to me. Which order do you mean? "
        "Sorry about that. OK then. Refunds are issued to the original payment method."
    )
    found = claims(text, [])
    assert [claim for claim, _ in found] == ["Refunds are issued to the original payment method."]
    assert claims("The zebra policy applies here today.", [r"zebra"]) == []


def test_tool_results_and_json_answers_count() -> None:
    answer = '{"answer": "Order A-1001 is shipped and totals 59.90 USD."}'
    case = make_case(
        input="Order A-1001?", expected={"answer": {"not_contains": ["zzz"]}, "groundedness": {}}
    )
    assert verdict(run_scripted(case, ORDER, answer), "groundedness") is True
    assert verdict(run_scripted(case, [], answer), "groundedness") is False
    assert prose(answer) == "Order A-1001 is shipped and totals 59.90 USD."
    assert prose("plain text") == "plain text" and prose("{broken") == "{broken"


def test_min_supported_turns_unsupported_claims_into_warnings() -> None:
    answer = (
        "Refunds are available within 30 days of delivery. Zebras prefer striped purple umbrellas."
    )
    strict = run_scripted(grounded_case(), SEARCH, answer)
    lenient = run_scripted(grounded_case(min_supported=0.5), SEARCH, answer)
    assert verdict(strict, "groundedness") is False and verdict(lenient, "groundedness") is True
    evaluation = lenient.evaluation("groundedness")
    assert evaluation is not None and evaluation.score == 0.5
    assert [item.severity for item in evaluation.findings] == ["warning"]


def test_evidence_helpers() -> None:
    assert (
        flatten({"order_id": "A-1", "items": [1, 2.5], "ok": True, "x": None})
        == "order id: A-1. items: 1. 2.5. ok: true"
    )
    case = make_case(instructions=[{"id": "sys", "text": "Be brief."}])
    result = run_scripted(
        case,
        [
            *SEARCH,
            ("get_order", {"order_id": "Z-1"}),
            ("send_email", {"to": "a@example.com", "subject": "s", "body": "hello"}),
        ],
        "x",
    )
    passages = collect(case, result.trace)
    sources = [passage.source for passage in passages]
    assert sources[:2] == ["request", "instruction:sys"] and "kb-refunds" in sources
    assert any("not_found" in passage.text for passage in passages), "error messages are evidence"
    assert not any("hello" in passage.text for passage in passages), (
        "what the agent sent is not evidence"
    )
    assert known_sources(result.trace) == frozenset({"kb-refunds"})
    assert "hello" in sink_text(result.trace) and "refunds" not in sink_text(result.trace)


# --------------------------------------------------------------------------- hallucination


@pytest.mark.parametrize(
    "output, expected_codes",
    [
        ("Order A-1001 is shipped and totals 59.90 USD.", []),
        ("Order A-1001 totals 59.9 USD, about 60 USD.", []),
        ("Order A-1001 totals 61.25 USD.", ["unsupported_number"]),
        ("Order A-1001 ships in 4 days.", ["unsupported_number"]),
        ("Order A-1001 has 2 parcels.", []),
        ("Order A-7777 is shipped.", ["unsupported_identifier"]),
        ("Order A-1001 was packed under policy POL-4821.", ["unsupported_identifier"]),
        ("Write to pat.lee@example.com about order A-1001.", []),
        ("Write to help@made-up.example about order A-1001.", ["unsupported_address"]),
        ("Track order A-1001 at https://track.made-up.example/A-1001", ["unsupported_address"]),
        ("Order A-1001: the tax rate is 0.15.", []),
        ("1. Order A-1001 is shipped.\n2. Nothing else.", []),
    ],
)
def test_atoms_need_a_source(output: str, expected_codes: list[str]) -> None:
    result = run_scripted(hallucination_case(), ORDER, output)
    assert codes(result, "hallucination") == expected_codes


def test_allow_list_and_switches() -> None:
    output = "Order A-1001 totals 61.25 USD under policy POL-4821."
    assert (
        codes(
            run_scripted(hallucination_case(allow=["61.25", "pol-4821"]), ORDER, output),
            "hallucination",
        )
        == []
    )
    numbers_off = run_scripted(hallucination_case(check_numbers=False), ORDER, output)
    assert codes(numbers_off, "hallucination") == ["unsupported_identifier"]
    identifiers_off = run_scripted(hallucination_case(check_identifiers=False), ORDER, output)
    assert codes(identifiers_off, "hallucination") == ["unsupported_number"]
    disabled = run_scripted(hallucination_case(enabled=False), ORDER, output)
    assert disabled.evaluation("hallucination") is None


def test_small_integers_are_checked_when_asked_or_when_a_unit_follows() -> None:
    output = "Order A-1001 has 4 parcels."
    assert codes(run_scripted(hallucination_case(), ORDER, output), "hallucination") == []
    strict = run_scripted(hallucination_case(ignore_integers_up_to=-1), ORDER, output)
    assert codes(strict, "hallucination") == ["unsupported_number"]
    for text in ("ready in 4 days", "a 5% fee", "takes 2 business days", "costs 9 USD"):
        result = run_scripted(hallucination_case(), ORDER, f"Order A-1001: {text}.")
        assert codes(result, "hallucination") == ["unsupported_number"], text


@pytest.mark.parametrize(
    "output, calls, flagged",
    [
        (
            "The refund has been issued for order A-1002.",
            [("issue_refund", {"order_id": "A-1002", "amount": 129.5})],
            False,
        ),
        ("The refund has been issued for order A-1002.", [], True),
        ("I have issued the refund.", [], True),
        (
            "Order A-1002 has been refunded.",
            [("issue_refund", {"order_id": "A-1002", "amount": 9999})],
            True,
        ),
        ("The refund could not be issued.", [], False),
        ("The refund has not been issued.", [], False),
        ("I will issue the refund once the order has been delivered.", [], False),
        ("A confirmation email has been sent to the customer.", [], True),
        ("I've emailed the customer.", [], True),
        ("Order A-1001 has been cancelled.", [], True),
        ("Order A-1001 is cancelled.", [], False),
        ("A ticket has been opened for the warehouse.", [], True),
    ],
)
def test_claimed_actions_need_a_successful_tool_call(
    output: str, calls: list, flagged: bool
) -> None:
    case = make_case(
        input="Refund order A-1002.",
        tools=["get_order", "issue_refund", "send_email", "cancel_order", "create_ticket"],
        expected={"answer": {"not_contains": ["zzz"]}, "hallucination": {}},
    )
    result = run_scripted(case, [("get_order", {"order_id": "A-1002"}), *calls], output)
    assert ("claimed_action_not_performed" in codes(result, "hallucination")) is flagged


def test_claimed_action_check_can_be_switched_off() -> None:
    case = make_case(
        expected={"answer": {"not_contains": ["zzz"]}, "hallucination": {"check_actions": False}}
    )
    assert codes(run_scripted(case, [], "The refund has been issued."), "hallucination") == []
