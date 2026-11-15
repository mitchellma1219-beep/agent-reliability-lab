"""Loading and validating suite files (golden datasets in YAML or JSON).

Suite files are data, not code. They are parsed with a restricted YAML loader: no object
construction, no aliases (which rules out "billion laughs" expansion) and no duplicate keys
(a repeated ``expected:`` would silently drop checks). Unknown fields are rejected by the
models.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .errors import ConfigError, SuiteError
from .instructions import same_priority_conflicts
from .matching import unknown_matchers
from .models import Case, Environment, Suite, ToolCallSpec
from .schema import check_schema
from .tools import BUILTIN_TOOLS, ToolSpec

MAX_SUITE_BYTES = 5_000_000
MAX_DEPTH = 40
SUFFIXES = {".yaml": "yaml", ".yml": "yaml", ".json": "json"}


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML with aliases and duplicate mapping keys turned into errors."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            line = getattr(self, "line", None)
            where = f" (near line {line + 1})" if isinstance(line, int) else ""
            raise yaml.constructor.ConstructorError(
                None, None, f"aliases and anchors are not allowed in suite files{where}", None
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    None, None, "mapping keys must be plain values", key_node.start_mark
                ) from exc
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
        return super().construct_mapping(node, deep)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _depth(value: Any) -> int:
    """Nesting depth, computed without recursion."""
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, level = stack.pop()
        deepest = max(deepest, level)
        if level > MAX_DEPTH:
            return level
        if isinstance(current, dict):
            stack.extend((item, level + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, level + 1) for item in current)
    return deepest


def parse_document(text: str, fmt: str = "yaml", source: str = "<string>") -> Any:
    """Parse YAML or JSON text into plain data, with the restrictions described above."""
    if len(text.encode("utf-8", errors="replace")) > MAX_SUITE_BYTES:
        raise SuiteError(f"{source}: larger than {MAX_SUITE_BYTES} bytes")
    try:
        if fmt == "json":
            data = json.loads(text, object_pairs_hook=_unique_pairs)
        else:
            loader = _StrictLoader(text)
            try:
                data = loader.get_single_data()
            finally:
                loader.dispose()
    except yaml.YAMLError as exc:
        raise SuiteError(f"{source}: invalid YAML: {str(exc)[:300]}") from exc
    except RecursionError as exc:
        raise SuiteError(f"{source}: nested too deeply") from exc
    except ValueError as exc:
        raise SuiteError(f"{source}: invalid JSON: {str(exc)[:300]}") from exc
    if _depth(data) > MAX_DEPTH:
        raise SuiteError(f"{source}: nested more than {MAX_DEPTH} levels deep")
    return data


def _describe(error: ValidationError, limit: int = 8) -> str:
    lines = []
    for item in error.errors()[:limit]:
        where = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"  {where}: {item['msg']}")
    extra = error.error_count() - limit
    if extra > 0:
        lines.append(f"  … and {extra} more")
    return "\n".join(lines)


def parse_suite(
    text: str,
    fmt: str = "yaml",
    source: str = "<string>",
    registry: Mapping[str, ToolSpec] | None = None,
) -> Suite:
    """Parse and fully validate a suite."""
    data = parse_document(text, fmt, source)
    if not isinstance(data, dict):
        raise SuiteError(f"{source}: a suite must be a mapping with 'name' and 'cases'")
    try:
        suite = Suite.model_validate(data)
    except ValidationError as exc:
        raise SuiteError(f"{source}: invalid suite\n{_describe(exc)}") from exc
    problems = validate_suite(suite, registry)
    if problems:
        shown = "\n".join(f"  {item}" for item in problems[:12])
        more = f"\n  … and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise SuiteError(f"{source}: invalid suite\n{shown}{more}")
    return suite


def load_suite(path: str | Path, registry: Mapping[str, ToolSpec] | None = None) -> Suite:
    """Load a suite from a ``.yaml``, ``.yml`` or ``.json`` file."""
    file = Path(path)
    fmt = SUFFIXES.get(file.suffix.lower())
    if fmt is None:
        raise SuiteError(f"{file}: suite files must end in .yaml, .yml or .json")
    if not file.is_file():
        raise SuiteError(f"{file}: no such suite file")
    if file.stat().st_size > MAX_SUITE_BYTES:
        raise SuiteError(f"{file}: larger than {MAX_SUITE_BYTES} bytes")
    try:
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SuiteError(f"{file}: cannot be read as UTF-8 text") from exc
    return parse_suite(text, fmt, str(file), registry)


# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------


def _merge_environment(base: Environment, extra: Environment) -> Environment:
    documents = {doc.id: doc for doc in base.documents}
    documents.update({doc.id: doc for doc in extra.documents})
    records = {name: dict(rows) for name, rows in base.records.items()}
    for name, rows in extra.records.items():
        records.setdefault(name, {}).update(rows)
    return Environment(
        documents=list(documents.values()),
        records=records,
        files={**base.files, **extra.files},
        api={**base.api, **extra.api},
    )


def resolve_case(suite: Suite, case: Case) -> Case:
    """Return the case with the suite defaults applied."""
    defaults = suite.defaults
    expected = case.expected.model_copy()
    if expected.hallucination is None and defaults.hallucination is not None:
        expected.hallucination = defaults.hallucination
    if defaults.budget is not None:
        expected.budget = (expected.budget or defaults.budget).merged_over(defaults.budget)
    return case.model_copy(
        update={
            "tools": list(case.tools if case.tools is not None else defaults.tools),
            "instructions": [*defaults.instructions, *case.instructions],
            "environment": _merge_environment(defaults.environment, case.environment),
            "canaries": list(dict.fromkeys([*defaults.canaries, *case.canaries])),
            "timeout_s": case.timeout_s if case.timeout_s is not None else defaults.timeout_s,
            "expected": expected,
        }
    )


def resolve_cases(suite: Suite) -> list[Case]:
    return [resolve_case(suite, case) for case in suite.cases]


# --------------------------------------------------------------------------------------
# Semantic validation
# --------------------------------------------------------------------------------------


def _call_specs(case: Case) -> list[tuple[str, ToolCallSpec]]:
    specs: list[tuple[str, ToolCallSpec]] = []
    if case.expected.tool_calls is not None:
        specs += [("expected.tool_calls.required", s) for s in case.expected.tool_calls.required]
        specs += [("expected.tool_calls.attempted", s) for s in case.expected.tool_calls.attempted]
        specs += [("expected.tool_calls.forbidden", s) for s in case.expected.tool_calls.forbidden]
    if case.attack is not None:
        specs += [("attack.indicators.tool_calls", s) for s in case.attack.indicators.tool_calls]
    return specs


def _validate_case(case: Case, registry: Mapping[str, ToolSpec]) -> list[str]:
    problems: list[str] = []
    available = set(case.tools or [])
    unknown = sorted(available - set(registry))
    if unknown:
        problems.append(f"unknown tool(s) {unknown}")
    for fault in case.faults:
        if fault.tool != "*" and fault.tool not in registry:
            problems.append(f"fault refers to unknown tool {fault.tool!r}")
        elif (
            fault.kind == "partial_response"
            and fault.tool != "*"
            and not registry[fault.tool].supports_partial
        ):
            problems.append(f"tool {fault.tool!r} cannot complete partially")
    for where, spec in _call_specs(case):
        if spec.tool not in registry:
            problems.append(f"{where} refers to unknown tool {spec.tool!r}")
        bad = unknown_matchers(spec.args)
        if bad:
            problems.append(f"{where} uses unknown matcher(s) {bad}")
    if case.expected.tool_calls is not None:
        for name in case.expected.tool_calls.order:
            if name not in registry:
                problems.append(f"expected.tool_calls.order refers to unknown tool {name!r}")
    if case.expected.workflow is not None:
        for step in case.expected.workflow.steps:
            if step.tool not in registry:
                problems.append(f"workflow step {step.id!r} refers to unknown tool {step.tool!r}")
            bad = unknown_matchers(step.args)
            if bad:
                problems.append(f"workflow step {step.id!r} uses unknown matcher(s) {bad}")
    if case.expected.answer is not None and case.expected.answer.json_fields is not None:
        bad = unknown_matchers(case.expected.answer.json_fields)
        if bad:
            problems.append(f"expected.answer.json_fields uses unknown matcher(s) {bad}")
    if case.sim is not None:
        for sim_step in case.sim.steps:
            if sim_step.tool not in available:
                problems.append(
                    f"sim step {sim_step.id!r} uses tool {sim_step.tool!r} that is not offered"
                )
    if case.output_schema is not None:
        try:
            check_schema(case.output_schema)
        except ConfigError as exc:
            problems.append(f"output_schema: {exc}")
    problems.extend(same_priority_conflicts(case.instructions))
    if case.expected.recovery is not None:
        recovery = case.expected.recovery
        if recovery.min_attempts is not None and recovery.min_attempts > recovery.max_attempts:
            problems.append("expected.recovery.min_attempts is greater than max_attempts")
    if not case.substantive_checks():
        problems.append(
            "has no substantive check: add expected.answer, tool_calls, workflow, recovery, "
            "an attack, an output_schema, instruction rules or a non-default status"
        )
    haystack = json.dumps(
        case.model_dump(mode="json", exclude={"canaries", "expected", "attack", "sim"}),
        ensure_ascii=False,
    )
    for canary in case.canaries:
        if json.dumps(canary, ensure_ascii=False)[1:-1] not in haystack:
            problems.append(f"canary {canary[:12]!r}… is not planted anywhere the agent can see")
    for name in case.environment.files:
        if name.startswith(("/", "\\")) or ".." in Path(name).parts or "\x00" in name:
            problems.append(f"unsafe fixture file name {name!r}")
    return problems


def validate_suite(suite: Suite, registry: Mapping[str, ToolSpec] | None = None) -> list[str]:
    """Semantic problems that the models alone cannot express. Empty when the suite is valid."""
    source = BUILTIN_TOOLS if registry is None else registry
    problems: list[str] = []
    for case in resolve_cases(suite):
        problems.extend(f"case {case.id!r}: {item}" for item in _validate_case(case, source))
    return problems
