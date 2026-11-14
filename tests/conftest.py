"""Shared fixtures. Everything is deterministic and offline."""

from __future__ import annotations

from typing import Any

import pytest

from agent_reliability_lab import load_builtin_suite, parse_suite
from agent_reliability_lab.models import Case, Suite


@pytest.fixture(scope="session")
def core_suite() -> Suite:
    return load_builtin_suite("core")


def make_case(**overrides: Any) -> Case:
    """A minimal valid case; keyword arguments replace top-level fields."""
    data: dict[str, Any] = {
        "id": "case-1",
        "category": "unit",
        "input": "What is the status of order A-1001?",
        "tools": ["get_order", "send_email", "calculator"],
        "environment": {
            "records": {
                "orders": {
                    "A-1001": {
                        "id": "A-1001",
                        "status": "shipped",
                        "total": 59.9,
                        "currency": "USD",
                        "email": "pat.lee@example.com",
                    }
                }
            }
        },
        "expected": {"answer": {"contains_all": ["shipped"]}},
    }
    data.update(overrides)
    return Case.model_validate(data)


def make_suite(*cases: dict[str, Any], **top: Any) -> Suite:
    """Build and validate a suite from case dictionaries (through the real loader)."""
    import json

    payload = {"name": "unit-suite", "cases": list(cases), **top}
    return parse_suite(json.dumps(payload), "json", "unit-suite")
