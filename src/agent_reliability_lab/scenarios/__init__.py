"""Built-in synthetic suites, shipped as YAML files next to this module.

Every fixture is invented. Addresses use reserved documentation domains (``example.com``,
``.example``, ``.test``, ``.invalid``); nothing here refers to a real system or person.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from ..errors import SuiteError
from ..loader import parse_suite, resolve_cases
from ..models import Suite

_PACKAGE = "agent_reliability_lab.scenarios"
ALL = "all"


def builtin_suite_names() -> list[str]:
    """Names of the bundled suites, in alphabetical order (without the combined ``all``)."""
    return sorted(
        entry.name[: -len(".yaml")]
        for entry in resources.files(_PACKAGE).iterdir()
        if entry.name.endswith(".yaml")
    )


@lru_cache(maxsize=32)
def _load(name: str) -> Suite:
    resource = resources.files(_PACKAGE).joinpath(f"{name}.yaml")
    return parse_suite(resource.read_text(encoding="utf-8"), "yaml", f"builtin:{name}")


@lru_cache(maxsize=1)
def _combined() -> Suite:
    cases = [case for name in builtin_suite_names() for case in resolve_cases(_load(name))]
    return Suite(
        name=ALL,
        version="+".join(_load(name).version for name in builtin_suite_names()),
        description="Every built-in suite: " + ", ".join(builtin_suite_names()) + ".",
        cases=cases,
    )


def load_builtin_suite(name: str) -> Suite:
    """Load a bundled suite by name (or ``all`` for every suite). Returns a fresh copy."""
    if name == ALL:
        return _combined().model_copy(deep=True)
    if name not in builtin_suite_names():
        choices = ", ".join([*builtin_suite_names(), ALL])
        raise SuiteError(f"no built-in suite named {name!r}; choose from {choices}")
    return _load(name).model_copy(deep=True)
