"""Matching recorded tool calls against expectations.

An expected argument is either a literal JSON value (compared for equality, numbers with a
small tolerance) or a matcher object whose keys all start with ``$``:

``{"$any": true}``            the argument is present
``{"$absent": true}``         the argument is not present
``{"$regex": "^A-\\d+$"}``    string matches the pattern (``re.search``)
``{"$contains": "refund"}``   string contains the text, case-insensitively
``{"$not_contains": "x"}``    string does not contain the text
``{"$in": [1, 2, 3]}``        value equals one of the listed values
``{"$approx": 59.9, "$tol": 0.01}``  number within the tolerance (default 1e-6)

``args`` is always a *subset* match: arguments that are not mentioned are not checked.
"""

from __future__ import annotations

import re
from typing import Any

from .models import ToolCall, ToolCallSpec

_ABSENT = object()
_MATCHERS = frozenset(
    {"$any", "$absent", "$regex", "$contains", "$not_contains", "$in", "$approx", "$tol"}
)


def is_matcher(value: Any) -> bool:
    return (
        isinstance(value, dict) and bool(value) and all(str(key).startswith("$") for key in value)
    )


def unknown_matchers(value: Any) -> list[str]:
    """Names of ``$``-keys that are not supported (used to reject typos when loading a suite)."""
    if is_matcher(value):
        return sorted(str(key) for key in value if key not in _MATCHERS)
    if isinstance(value, dict):
        return sorted({name for item in value.values() for name in unknown_matchers(item)})
    if isinstance(value, list):
        return sorted({name for item in value for name in unknown_matchers(item)})
    return []


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def values_equal(expected: Any, actual: Any) -> bool:
    """Deep equality with a tolerance for numbers; ``True`` is not equal to ``1``."""
    if _number(expected) and _number(actual):
        return abs(float(expected) - float(actual)) <= 1e-9 * max(1.0, abs(float(expected)))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            values_equal(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            values_equal(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return bool(expected == actual)


def matches(expected: Any, actual: Any = _ABSENT) -> bool:
    """Does ``actual`` satisfy ``expected`` (a literal or a ``$``-matcher)?"""
    if not is_matcher(expected):
        if actual is _ABSENT:
            return False
        if isinstance(expected, dict) and isinstance(actual, dict):
            return all(matches(item, actual.get(key, _ABSENT)) for key, item in expected.items())
        return values_equal(expected, actual)
    if expected.get("$absent"):
        return actual is _ABSENT
    if actual is _ABSENT:
        return False
    if "$regex" in expected and not (
        isinstance(actual, str) and re.search(str(expected["$regex"]), actual)
    ):
        return False
    if "$contains" in expected and not (
        isinstance(actual, str) and str(expected["$contains"]).casefold() in actual.casefold()
    ):
        return False
    if "$not_contains" in expected and (
        isinstance(actual, str) and str(expected["$not_contains"]).casefold() in actual.casefold()
    ):
        return False
    if "$in" in expected and not any(values_equal(item, actual) for item in expected["$in"]):
        return False
    if "$approx" in expected:
        tolerance = float(expected.get("$tol", 1e-6))
        if not (_number(actual) and abs(float(actual) - float(expected["$approx"])) <= tolerance):
            return False
    return True


def args_match(expected: dict[str, Any], arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return not expected
    return all(matches(item, arguments.get(key, _ABSENT)) for key, item in expected.items())


def call_matches(spec: ToolCallSpec, call: ToolCall) -> bool:
    return call.tool == spec.tool and args_match(spec.args, call.arguments)
