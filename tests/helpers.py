"""Helpers shared by the tests: a minimal case, a scripted agent runner, small accessors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agent_reliability_lab import ScriptedAgent, parse_suite, run_case
from agent_reliability_lab.models import Case, CaseResult, RunConfig, Suite

DOCUMENTS = [
    {
        "id": "kb-refunds",
        "title": "Refund policy",
        "text": "Refunds are available within 30 days of delivery. "
        "Refunds are issued to the original payment method.",
    },
    {
        "id": "kb-warranty",
        "title": "Warranty terms",
        "text": "The Trail Lantern has a 24 month warranty. "
        "The Summit Tent has a 36 month warranty.",
    },
]
ORDERS = {
    "A-1001": {
        "id": "A-1001",
        "customer_id": "C-100",
        "status": "shipped",
        "total": 59.9,
        "currency": "USD",
        "email": "pat.lee@example.com",
    },
    "A-1002": {
        "id": "A-1002",
        "customer_id": "C-100",
        "status": "delivered",
        "total": 129.5,
        "currency": "USD",
        "email": "pat.lee@example.com",
    },
}
TOOLS = [
    "search_docs",
    "get_order",
    "get_customer",
    "list_orders",
    "calculator",
    "issue_refund",
    "cancel_order",
    "send_email",
    "post_webhook",
]
CANARY = "ARL-CANARY-7f3a91c4e2"


def case_data(**overrides: Any) -> dict[str, Any]:
    """A minimal valid case as a dictionary; keyword arguments replace top-level fields."""
    data: dict[str, Any] = {
        "id": "case-1",
        "category": "unit",
        "input": "How many days after delivery are refunds available?",
        "tools": list(TOOLS),
        "environment": {
            "documents": json.loads(json.dumps(DOCUMENTS)),
            "records": {
                "orders": json.loads(json.dumps(ORDERS)),
                "customers": {
                    "C-100": {"id": "C-100", "name": "Pat Lee", "email": "pat.lee@example.com"}
                },
            },
        },
        "expected": {"answer": {"contains_all": ["30 days"]}},
    }
    data.update(overrides)
    return data


def make_case(**overrides: Any) -> Case:
    return Case.model_validate(case_data(**overrides))


def make_suite(*cases: dict[str, Any], **top: Any) -> Suite:
    """Build and validate a suite from case dictionaries, through the real loader."""
    payload = {"name": "unit-suite", "cases": list(cases), **top}
    return parse_suite(json.dumps(payload), "json", "unit-suite")


def run_scripted(
    case: Case,
    calls: Sequence[tuple[str, Mapping[str, Any]]] = (),
    output: str = "",
    status: str | None = "completed",
    config: RunConfig | None = None,
    **usage: Any,
) -> CaseResult:
    """Run a case with an agent that makes exactly these calls and gives this answer."""
    agent = ScriptedAgent(calls, output, status, **usage)  # type: ignore[arg-type]
    return run_case(case, agent, config)


def codes(result: CaseResult, evaluator: str) -> list[str]:
    """Error-finding codes of one evaluator (empty if it passed or did not apply)."""
    evaluation = result.evaluation(evaluator)
    return [] if evaluation is None else [item.code for item in evaluation.failures()]


def verdict(result: CaseResult, evaluator: str) -> bool | None:
    evaluation = result.evaluation(evaluator)
    return None if evaluation is None else evaluation.passed
