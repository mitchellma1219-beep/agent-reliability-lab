"""JSON Schema validation that never touches the network.

``jsonschema`` can be asked to fetch a remote ``$ref``. Schemas here come from suite files and
tool definitions, so every validator is built with an empty reference registry: a remote or
unknown ``$ref`` is reported as a validation error instead of being downloaded.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import Unresolvable

from .errors import ConfigError

MAX_ERRORS = 10
_EMPTY_REGISTRY: Registry[Any] = Registry()


def check_schema(schema: Any) -> None:
    """Raise :class:`ConfigError` if ``schema`` is not a valid Draft 2020-12 schema."""
    if not isinstance(schema, dict):
        raise ConfigError("a JSON Schema must be an object")
    try:
        Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ConfigError(f"invalid JSON Schema: {exc.message}") from exc


def build_validator(schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema, registry=_EMPTY_REGISTRY)


def validate(schema: dict[str, Any], instance: Any, *, validator: Any = None) -> list[str]:
    """Return human-readable violations (at most :data:`MAX_ERRORS`), empty if valid."""
    active = validator if validator is not None else build_validator(schema)
    problems: list[str] = []
    try:
        for error in active.iter_errors(instance):
            location = "/".join(str(part) for part in error.absolute_path) or "(root)"
            problems.append(f"{location}: {error.message[:300]}")
            if len(problems) >= MAX_ERRORS:
                break
    except Unresolvable as exc:
        problems.append(f"(schema): unresolvable $ref {getattr(exc, 'ref', '')!s:.200}")
    except RecursionError:
        problems.append("(root): value is nested too deeply to validate")
    return problems


@lru_cache(maxsize=256)
def _cached_validator(schema_json: str) -> Draft202012Validator:
    import json

    return build_validator(json.loads(schema_json))


def cached_validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Validator cache keyed by the schema's canonical JSON text."""
    import json

    return _cached_validator(json.dumps(schema, sort_keys=True))
