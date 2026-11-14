"""Shared fixtures. Everything is deterministic and offline."""

from __future__ import annotations

import pytest

from agent_reliability_lab import guard, load_builtin_suite
from agent_reliability_lab.models import Suite

# CPython does not trace audit hooks unless the hook opts in, so coverage would report the
# guard's hook as never executed although the tests exercise it. Opting in here, for the test
# process only, lets the coverage tool see that code. It does not change what the hook does.
guard._hook.__cantrace__ = True  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def core_suite() -> Suite:
    return load_builtin_suite("core")


@pytest.fixture(scope="session")
def all_suite() -> Suite:
    return load_builtin_suite("all")
