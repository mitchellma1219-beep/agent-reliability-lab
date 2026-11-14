"""Typed models for suites, cases, traces and results.

Every model forbids unknown fields. In a testing tool a misspelt key such as ``forbiden:``
must be an error, otherwise the check it was meant to configure silently never runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

SCHEMA_VERSION = 1

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
MAX_TEXT_CHARS = 200_000
MAX_PATTERN_CHARS = 500

Status = Literal["completed", "needs_clarification", "refused", "abstained", "failed"]
STATUSES: tuple[Status, ...] = (
    "completed",
    "needs_clarification",
    "refused",
    "abstained",
    "failed",
)

FaultKind = Literal[
    "timeout",
    "rate_limit",
    "transient_error",
    "permanent_error",
    "incorrect_response",
    "partial_response",
    "invalid_output",
    "retry_exhaustion",
    "dependency_failure",
]


class Model(BaseModel):
    """Base model: unknown fields are rejected, assignments are validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _check_pattern(pattern: str) -> str:
    if len(pattern) > MAX_PATTERN_CHARS:
        raise ValueError(f"regular expression longer than {MAX_PATTERN_CHARS} characters")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from exc
    return pattern


# --------------------------------------------------------------------------------------
# What the agent is given
# --------------------------------------------------------------------------------------


class Rule(Model):
    """Machine-checkable form of one instruction.

    The same rule is shown to the agent (as part of the task) and used by the
    ``instruction_following`` evaluator, so there is one statement of what was asked.
    """

    max_words: int | None = Field(default=None, ge=1)
    min_words: int | None = Field(default=None, ge=1)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    confidential: list[str] = Field(default_factory=list)
    must_match: str | None = None
    must_not_match: str | None = None
    format: Literal["json", "bullets", "numbered", "plain"] | None = None
    starts_with: str | None = None
    ends_with: str | None = None
    cite_sources: bool = False

    @field_validator("must_match", "must_not_match")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        return None if value is None else _check_pattern(value)

    def is_empty(self) -> bool:
        return self == Rule()


class Instruction(Model):
    """One instruction given to the agent, with an optional machine-checkable rule."""

    id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    priority: Literal["system", "user"] = "system"
    rule: Rule | None = None


class Document(Model):
    """A document in the retrieval corpus or in the context given up front."""

    id: str = Field(pattern=ID_PATTERN)
    title: str = Field(default="", max_length=500)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class FillerSpec(Model):
    """Deterministically generated distractor documents for long-context cases."""

    count: int = Field(ge=1, le=500)
    words: int = Field(default=80, ge=10, le=2000)
    seed: int = 0


class Environment(Model):
    """The simulated world a case runs in. Everything here is synthetic fixture data."""

    documents: list[Document] = Field(default_factory=list)
    records: dict[str, dict[str, dict[str, JsonValue]]] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    api: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("documents")
    @classmethod
    def _unique_documents(cls, value: list[Document]) -> list[Document]:
        seen: set[str] = set()
        for doc in value:
            if doc.id in seen:
                raise ValueError(f"duplicate document id {doc.id!r}")
            seen.add(doc.id)
        return value


class Mutation(Model):
    """How an ``incorrect_response`` or ``invalid_output`` fault changes a tool result."""

    op: Literal[
        "scale", "negate", "set", "drop_field", "truncate_json", "wrong_type", "null", "stringify"
    ]
    field: str | None = None
    factor: float | None = None
    value: JsonValue = None

    @model_validator(mode="after")
    def _consistent(self) -> Mutation:
        if self.op in {"scale", "negate", "set", "drop_field"} and not self.field:
            raise ValueError(f"mutation {self.op!r} needs a 'field'")
        if self.op == "scale" and self.factor is None:
            raise ValueError("mutation 'scale' needs a 'factor'")
        return self


RESULT_MUTATIONS = frozenset({"scale", "negate", "set", "drop_field"})
OUTPUT_MUTATIONS = frozenset({"truncate_json", "wrong_type", "null", "stringify"})


class FaultSpec(Model):
    """One injected fault. ``times=None`` means every matching call."""

    kind: FaultKind
    tool: str = "*"
    dependency: str | None = None
    times: int | None = Field(default=None, ge=1)
    after: int = Field(default=0, ge=0)
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    retry_after_ms: float = Field(default=1000.0, ge=0.0)
    delay_ms: float | None = Field(default=None, ge=0.0)
    mutation: Mutation | None = None

    @model_validator(mode="after")
    def _consistent(self) -> FaultSpec:
        if self.kind == "dependency_failure" and not self.dependency:
            raise ValueError("a dependency_failure fault needs a 'dependency'")
        if self.kind != "dependency_failure" and self.dependency:
            raise ValueError("'dependency' is only valid for dependency_failure faults")
        if self.kind == "retry_exhaustion" and self.times is not None:
            raise ValueError("retry_exhaustion affects every call; remove 'times'")
        if self.kind == "incorrect_response":
            if self.mutation is None or self.mutation.op not in RESULT_MUTATIONS:
                raise ValueError(
                    "incorrect_response needs a mutation with op in "
                    + str(sorted(RESULT_MUTATIONS))
                )
        elif self.kind == "invalid_output":
            if self.mutation is not None and self.mutation.op not in OUTPUT_MUTATIONS:
                raise ValueError(
                    "invalid_output accepts a mutation with op in " + str(sorted(OUTPUT_MUTATIONS))
                )
        elif self.mutation is not None:
            raise ValueError(f"'mutation' is not valid for a {self.kind} fault")
        return self


# --------------------------------------------------------------------------------------
# Hints for the bundled simulated agents (never read by evaluators)
# --------------------------------------------------------------------------------------


class SimCondition(Model):
    """Guard on a simulated step: run it only if a value has (or does not have) a given value."""

    ref: str
    equals: JsonValue = None
    one_of: list[JsonValue] | None = None
    not_equals: JsonValue = None


class SimStep(Model):
    """One tool call a careful reader of the request would make."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    tool: str
    args: dict[str, JsonValue] = Field(default_factory=dict)
    when: SimCondition | None = None


class SimRead(Model):
    """Answer by quoting the best-matching sentence from retrieved or provided documents."""

    question: str | None = None
    from_steps: list[str] = Field(default_factory=list)
    use_context: bool = True


class SimOtherwise(Model):
    """What to say when a guard is false (for example, a refund that is not allowed)."""

    answer: str
    status: Status = "refused"


class SimInjectedCall(Model):
    tool: str
    args: dict[str, JsonValue] = Field(default_factory=dict)


class SimInjection(Model):
    """What an injected instruction asks for, as a gullible agent would read it.

    ``marker`` is a substring of the injected text. The gullible simulated agent only acts
    if that text actually reached it (in the request, a retrieved document or a tool result).
    """

    marker: str = Field(min_length=3)
    calls: list[SimInjectedCall] = Field(default_factory=list)
    say: str | None = None


class SimPlan(Model):
    """A structured reading of the request, used only by the bundled simulated agents.

    Simulated agents have no language understanding, so each bundled case states what the
    request asks for. Real agents ignore this block and evaluators never read it; the oracle
    lives in ``expected``, which agents never see.
    """

    steps: list[SimStep] = Field(default_factory=list)
    answer: str | None = None
    answer_json: JsonValue = None
    read: SimRead | None = None
    otherwise: SimOtherwise | None = None
    missing: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    guess: dict[str, JsonValue] = Field(default_factory=dict)
    injections: list[SimInjection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> SimPlan:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("sim step ids must be unique")
        if self.missing and not self.clarifying_question:
            raise ValueError("sim.missing needs sim.clarifying_question")
        return self


# --------------------------------------------------------------------------------------
# The oracle: what evaluators check (never shown to agents)
# --------------------------------------------------------------------------------------


class NumericCheck(Model):
    value: float
    tolerance: float = Field(default=1e-9, ge=0.0)


class AnswerCheck(Model):
    """Deterministic checks on the final output."""

    contains_all: list[str] = Field(default_factory=list)
    contains_any: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    regex: list[str] = Field(default_factory=list)
    equals: str | None = None
    numeric: NumericCheck | None = None
    json_fields: dict[str, JsonValue] | None = None
    case_sensitive: bool = False

    @field_validator("regex")
    @classmethod
    def _valid_regex(cls, value: list[str]) -> list[str]:
        return [_check_pattern(item) for item in value]

    def is_empty(self) -> bool:
        return self == AnswerCheck() or self == AnswerCheck(case_sensitive=True)


class ToolCallSpec(Model):
    """A tool call to look for. ``args`` is a subset match and may use ``$``-matchers."""

    tool: str
    args: dict[str, JsonValue] = Field(default_factory=dict)
    min_count: int = Field(default=1, ge=1)
    max_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> ToolCallSpec:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be >= min_count")
        return self


class ToolCallExpectation(Model):
    required: list[ToolCallSpec] = Field(default_factory=list)
    forbidden: list[ToolCallSpec] = Field(default_factory=list)
    order: list[str] = Field(default_factory=list)
    allow_extra: bool = True
    max_calls: int | None = Field(default=None, ge=0)


class GroundednessCheck(Model):
    """Lexical support of each output sentence in the evidence the agent actually saw."""

    enabled: bool = True
    token_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    passage_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_supported: float = Field(default=1.0, ge=0.0, le=1.0)
    ignore_patterns: list[str] = Field(default_factory=list)

    @field_validator("ignore_patterns")
    @classmethod
    def _valid_regex(cls, value: list[str]) -> list[str]:
        return [_check_pattern(item) for item in value]


class HallucinationCheck(Model):
    """Specifics in the output (numbers, identifiers, addresses, claimed actions) need evidence."""

    enabled: bool = True
    allow: list[str] = Field(default_factory=list)
    check_numbers: bool = True
    check_identifiers: bool = True
    check_actions: bool = True
    ignore_integers_up_to: int = Field(default=10, ge=-1)


class WorkflowStep(Model):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    tool: str
    args: dict[str, JsonValue] = Field(default_factory=dict)
    after: list[str] = Field(default_factory=list)


class StateAssertion(Model):
    """A check on the simulated world after the run, addressed by a path of keys and indexes."""

    path: list[str | int] = Field(min_length=1)
    equals: JsonValue = None
    contains: str | None = None
    length: int | None = Field(default=None, ge=0)
    exists: bool | None = None

    @model_validator(mode="after")
    def _one_check(self) -> StateAssertion:
        chosen = [
            "equals" in self.model_fields_set,
            self.contains is not None,
            self.length is not None,
            self.exists is not None,
        ]
        if sum(chosen) != 1:
            raise ValueError("a state assertion needs exactly one of equals/contains/length/exists")
        return self


class WorkflowExpectation(Model):
    steps: list[WorkflowStep] = Field(default_factory=list)
    final_state: list[StateAssertion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> WorkflowExpectation:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("workflow step ids must be unique")
        known = set(ids)
        for step in self.steps:
            unknown = [dep for dep in step.after if dep not in known]
            if unknown:
                raise ValueError(f"workflow step {step.id!r} depends on unknown step(s) {unknown}")
            if step.id in step.after:
                raise ValueError(f"workflow step {step.id!r} depends on itself")
        if not self.steps and not self.final_state:
            raise ValueError("a workflow expectation needs steps or final_state assertions")
        return self


class RecoveryExpectation(Model):
    """How the agent should cope with injected faults."""

    outcome: Literal["recovered", "failed_safely"]
    min_attempts: int | None = Field(default=None, ge=1)
    max_attempts: int = Field(default=4, ge=1)
    respect_retry_after: bool = True


class Budget(Model):
    max_latency_ms: float | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, ge=0)

    def merged_over(self, base: Budget | None) -> Budget:
        """Fields set here win; unset fields fall back to ``base``."""
        if base is None:
            return self
        data = base.model_dump()
        data.update({key: value for key, value in self.model_dump().items() if value is not None})
        return Budget(**data)


class AttackIndicators(Model):
    """Observable signs that an injected instruction was obeyed."""

    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    output_contains: list[str] = Field(default_factory=list)
    sink_contains: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _non_empty(self) -> AttackIndicators:
        if not (self.tool_calls or self.output_contains or self.sink_contains):
            raise ValueError("an attack needs at least one success indicator")
        return self


class Attack(Model):
    kind: Literal["direct", "indirect_document", "indirect_tool_response"]
    goal: str = Field(min_length=1, max_length=500)
    indicators: AttackIndicators


class Expected(Model):
    """Everything the evaluators compare the trace against."""

    status: Status | list[Status] = "completed"
    answer: AnswerCheck | None = None
    tool_calls: ToolCallExpectation | None = None
    groundedness: GroundednessCheck | None = None
    hallucination: HallucinationCheck | None = None
    workflow: WorkflowExpectation | None = None
    recovery: RecoveryExpectation | None = None
    budget: Budget | None = None

    def statuses(self) -> tuple[Status, ...]:
        return (self.status,) if isinstance(self.status, str) else tuple(self.status)

    @field_validator("status")
    @classmethod
    def _non_empty_status(cls, value: Status | list[Status]) -> Status | list[Status]:
        if isinstance(value, list) and not value:
            raise ValueError("status must name at least one acceptable status")
        return value


# --------------------------------------------------------------------------------------
# Cases and suites
# --------------------------------------------------------------------------------------


class Case(Model):
    """One test case of a golden dataset."""

    id: str = Field(pattern=ID_PATTERN)
    category: str = Field(pattern=r"^[a-z][a-z0-9_]{0,49}$")
    description: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list)
    input: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    instructions: list[Instruction] = Field(default_factory=list)
    context: list[Document] = Field(default_factory=list)
    context_filler: FillerSpec | None = None
    environment: Environment = Field(default_factory=Environment)
    tools: list[str] | None = None
    faults: list[FaultSpec] = Field(default_factory=list)
    canaries: list[str] = Field(default_factory=list)
    attack: Attack | None = None
    output_schema: dict[str, Any] | None = None
    sim: SimPlan | None = None
    expected: Expected = Field(default_factory=Expected)
    timeout_s: float | None = Field(default=None, gt=0, le=600)

    @field_validator("canaries")
    @classmethod
    def _canaries_long_enough(cls, value: list[str]) -> list[str]:
        for canary in value:
            if len(canary) < 8:
                raise ValueError("canary strings must be at least 8 characters long")
        return value

    @field_validator("instructions")
    @classmethod
    def _unique_instructions(cls, value: list[Instruction]) -> list[Instruction]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("instruction ids must be unique within a case")
        return value

    def has_rules(self) -> bool:
        return any(item.rule is not None and not item.rule.is_empty() for item in self.instructions)

    def substantive_checks(self) -> list[str]:
        """Names of the checks that can actually tell a good run from a bad one."""
        expected = self.expected
        checks: list[str] = []
        if expected.statuses() != ("completed",):
            checks.append("status")
        if expected.answer is not None and not expected.answer.is_empty():
            checks.append("answer")
        if expected.tool_calls is not None:
            checks.append("tool_calls")
        if expected.groundedness is not None and expected.groundedness.enabled:
            checks.append("groundedness")
        if expected.workflow is not None:
            checks.append("workflow")
        if expected.recovery is not None:
            checks.append("recovery")
        if self.output_schema is not None:
            checks.append("output_schema")
        if self.attack is not None:
            checks.append("attack")
        if self.has_rules():
            checks.append("instructions")
        return checks


class SuiteDefaults(Model):
    """Values shared by every case of a suite. Cases add to or override them."""

    tools: list[str] = Field(default_factory=list)
    instructions: list[Instruction] = Field(default_factory=list)
    environment: Environment = Field(default_factory=Environment)
    canaries: list[str] = Field(default_factory=list)
    hallucination: HallucinationCheck | None = None
    budget: Budget | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=600)


class Suite(Model):
    """A named golden dataset."""

    name: str = Field(pattern=ID_PATTERN)
    version: str = Field(default="1", max_length=40)
    description: str = Field(default="", max_length=4000)
    defaults: SuiteDefaults = Field(default_factory=SuiteDefaults)
    cases: list[Case] = Field(min_length=1, max_length=5000)

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, value: list[Case]) -> list[Case]:
        seen: set[str] = set()
        for case in value:
            if case.id in seen:
                raise ValueError(f"duplicate case id {case.id!r}")
            seen.add(case.id)
        return value

    def categories(self) -> list[str]:
        return sorted({case.category for case in self.cases})


class AgentTask(Model):
    """What an agent is allowed to see of a case. There is deliberately no ``expected``."""

    input: str
    instructions: list[Instruction] = Field(default_factory=list)
    context: list[Document] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    sim: SimPlan | None = None


# --------------------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------------------


class ToolErrorInfo(Model):
    code: str
    message: str
    retryable: bool = False
    retry_after_ms: float | None = None


class ToolCall(Model):
    """One call to a simulated tool, recorded by the harness (not by the agent)."""

    index: int
    tool: str
    arguments: JsonValue = None
    ok: bool
    result: JsonValue = None
    result_truncated: bool = False
    error: ToolErrorInfo | None = None
    attempt: int = 1
    is_retry: bool = False
    latency_ms: float = 0.0
    started_ms: float = 0.0
    fault: str | None = None
    sink: bool = False
    side_effect: bool = False


class LlmCall(Model):
    """One model call reported by the agent or its adapter."""

    index: int
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    started_ms: float = 0.0
    note: str = ""


class Wait(Model):
    """A pause the agent took, for example backing off before a retry."""

    index: int
    ms: float = Field(ge=0.0)
    started_ms: float = 0.0
    reason: str = ""


class SecurityEvent(Model):
    """Something the containment layer blocked or noticed."""

    kind: str
    detail: str


class TraceError(Model):
    kind: Literal["agent_exception", "agent_exit", "timeout", "harness_abort", "invalid_output"]
    message: str


class ModelUsage(Model):
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class Usage(Model):
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    by_model: dict[str, ModelUsage] = Field(default_factory=dict)


class Trace(Model):
    """Everything observed while one agent handled one case once."""

    case_id: str
    trial: int
    seed: int
    agent_name: str
    agent_version: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    llm_calls: list[LlmCall] = Field(default_factory=list)
    waits: list[Wait] = Field(default_factory=list)
    output: str = ""
    output_truncated: bool = False
    status: Status = "failed"
    status_source: Literal["declared", "inferred", "harness"] = "harness"
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    wall_ms: float = 0.0
    errors: list[TraceError] = Field(default_factory=list)
    security_events: list[SecurityEvent] = Field(default_factory=list)
    final_state: dict[str, JsonValue] = Field(default_factory=dict)
    agent_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def retries(self) -> int:
        return sum(1 for call in self.tool_calls if call.is_retry)

    def successful_calls(self, tool: str | None = None) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.ok and (tool is None or c.tool == tool)]

    def steps(self) -> list[ToolCall | LlmCall | Wait]:
        """All recorded steps in the order they happened."""
        merged: list[ToolCall | LlmCall | Wait] = [*self.tool_calls, *self.llm_calls, *self.waits]
        return sorted(merged, key=lambda step: step.index)


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


class Finding(Model):
    code: str
    message: str
    severity: Literal["error", "warning", "info"] = "error"


class Evaluation(Model):
    """The verdict of one evaluator on one trace."""

    evaluator: str
    passed: bool
    score: float | None = None
    findings: list[Finding] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)

    def failures(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]


class CaseResult(Model):
    """One case, one trial: the trace plus every applicable evaluation."""

    case_id: str
    category: str
    trial: int
    seed: int
    passed: bool
    evaluations: list[Evaluation] = Field(default_factory=list)
    failure_categories: list[str] = Field(default_factory=list)
    estimated_cost_usd: float | None = None
    trace: Trace

    def evaluation(self, name: str) -> Evaluation | None:
        for item in self.evaluations:
            if item.evaluator == name:
                return item
        return None

    def failed_evaluators(self) -> list[str]:
        return [item.evaluator for item in self.evaluations if not item.passed]


class ModelPrice(Model):
    """Price per million tokens. Supplied by the user; never fetched."""

    input_per_mtok: float = Field(ge=0.0)
    output_per_mtok: float = Field(ge=0.0)


class PriceTable(Model):
    """Prices used to turn token counts into an *estimated* cost."""

    name: str = "illustrative"
    currency: str = "USD"
    models: dict[str, ModelPrice] = Field(default_factory=dict)
    default: ModelPrice | None = None

    def estimate(self, usage: Usage) -> float | None:
        """Estimated cost, or ``None`` if any model that was used has no price."""
        total = 0.0
        for model, used in usage.by_model.items():
            price = self.models.get(model, self.default)
            if price is None:
                return None
            total += used.input_tokens / 1e6 * price.input_per_mtok
            total += used.output_tokens / 1e6 * price.output_per_mtok
        return total


def default_price_table() -> PriceTable:
    """Illustrative prices for the bundled simulated model. Not any vendor's real prices."""
    return PriceTable(
        name="illustrative",
        models={"sim-1": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0)},
    )


class GuardConfig(Model):
    """In-process tripwire applied to the agent's thread. Not a security boundary."""

    enabled: bool = True
    allow_network: bool = False
    allow_subprocess: bool = False
    strict_reads: bool = False
    extra_read_roots: list[str] = Field(default_factory=list)


class RunConfig(Model):
    seed: int = 0
    trials: int = Field(default=1, ge=1, le=1000)
    timeout_s: float = Field(default=10.0, gt=0, le=600)
    max_tool_calls: int = Field(default=50, ge=1, le=10_000)
    max_llm_calls: int = Field(default=200, ge=1, le=100_000)
    max_output_chars: int = Field(default=100_000, ge=100, le=10_000_000)
    extra_faults: list[FaultSpec] = Field(default_factory=list)
    fault_profile: str | None = None
    prices: PriceTable = Field(default_factory=default_price_table)
    guard: GuardConfig = Field(default_factory=GuardConfig)


class RunResult(Model):
    """A stored run: one agent version against one suite."""

    schema_version: int = SCHEMA_VERSION
    run_id: str = Field(pattern=ID_PATTERN)
    created_at: str
    arl_version: str
    suite_name: str
    suite_version: str
    suite_description: str = ""
    agent_name: str
    agent_version: str
    config: RunConfig
    results: list[CaseResult] = Field(default_factory=list)
    notes: str = ""

    def case_ids(self) -> list[str]:
        """Case ids in first-seen order."""
        return list(dict.fromkeys(result.case_id for result in self.results))

    def by_case(self) -> dict[str, list[CaseResult]]:
        grouped: dict[str, list[CaseResult]] = {}
        for result in self.results:
            grouped.setdefault(result.case_id, []).append(result)
        return grouped

    def fingerprint(self) -> str:
        """Hash of everything that is supposed to be reproducible.

        Wall-clock measurements, the creation time and the run id are left out, so two runs
        of a deterministic agent with the same seed must have the same fingerprint.
        """
        payload = self.model_dump(mode="json", exclude={"run_id", "created_at", "notes"})
        for result in payload["results"]:
            result["trace"].pop("wall_ms", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
