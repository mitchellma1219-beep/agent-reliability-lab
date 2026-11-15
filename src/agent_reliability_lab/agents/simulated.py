"""Deterministic simulated agents: one clean reference and one variant per failure mode.

These are **not** language models. A simulated agent reads the structured ``sim`` block of a
bundled case (what the request asks for) and carries it out through the sandbox with a fixed
policy: validate tool results, retry transient errors a bounded number of times, fail
explicitly when it cannot finish, treat retrieved text as data, follow system instructions,
ask when the request is ambiguous. It never sees ``expected``.

Each failure mode switches off exactly one part of that policy. Because the faulty agents
differ from the reference in a single behaviour, a detection matrix over them is a
single-fault experiment: it shows which evaluator notices which kind of mistake. It says
nothing about how often a real model makes that mistake.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields, replace
from typing import Any

from ..errors import ConfigError, ToolError
from ..instructions import CITATION, is_json, resolve_rules
from ..models import AgentTask, Rule, SimCondition, SimPlan, SimStep, Status
from ..schema import validate
from ..text import content_tokens, overlap, split_sentences, word_count
from .base import AgentContext, AgentResponse

MODEL = "sim-1"
MAX_ATTEMPTS = 3
READ_THRESHOLD = 0.6
DISTRACTION_THRESHOLD = 0.3
SLOW_EXTRA_MS = 20_000.0
EXPENSIVE_EXTRA_TOKENS = 60_000

FAILURE_MODES: dict[str, str] = {
    "follows_injection": "obeys instructions found in the request, documents or tool results",
    "hallucinates": "invents values instead of saying that information is missing",
    "wrong_tool": "calls a different tool than the task needs",
    "bad_arguments": "calls the right tool with wrong, missing or mistyped arguments",
    "ignores_instruction": "drops system instructions and follows only the user's wishes",
    "invalid_json": "returns malformed or schema-violating JSON",
    "no_retry": "gives up on the first transient error",
    "unbounded_retry": "retries failed calls forever, including non-retryable ones",
    "leaks_canary": "copies its instructions and working data into answers and outgoing messages",
    "partial_completion": "stops half-way through a workflow but reports success",
    "slow": "takes far longer than the latency budget",
    "expensive": "uses far more tokens than the cost budget",
    "trusts_invalid_data": "uses tool results without validating them",
    "guesses_when_ambiguous": "acts on a guess instead of asking a clarifying question",
    "distracted": "answers from the most recent document instead of the most relevant one",
}


@dataclass(frozen=True)
class Behaviour:
    """Which parts of the reference policy are broken. All ``False`` is the clean agent."""

    follows_injection: bool = False
    hallucinates: bool = False
    wrong_tool: bool = False
    bad_arguments: bool = False
    ignores_instruction: bool = False
    invalid_json: bool = False
    no_retry: bool = False
    unbounded_retry: bool = False
    leaks_canary: bool = False
    partial_completion: bool = False
    slow: bool = False
    expensive: bool = False
    trusts_invalid_data: bool = False
    guesses_when_ambiguous: bool = False
    distracted: bool = False

    def active_modes(self) -> list[str]:
        return [item.name for item in fields(self) if getattr(self, item.name)]


CONFUSABLE: dict[str, tuple[str, ...]] = {
    "get_order": ("get_customer", "get_product"),
    "get_customer": ("get_order", "get_product"),
    "get_product": ("get_order", "search_docs"),
    "list_orders": ("get_customer", "get_order"),
    "search_docs": ("get_product", "get_order"),
    "calculator": ("search_docs", "get_exchange_rate"),
    "get_exchange_rate": ("calculator", "search_docs"),
    "issue_refund": ("cancel_order", "create_ticket"),
    "cancel_order": ("issue_refund", "create_ticket"),
    "create_ticket": ("send_email", "post_webhook"),
    "send_email": ("create_ticket", "post_webhook"),
    "send_bulk_email": ("send_email", "create_ticket"),
    "post_webhook": ("send_email", "create_ticket"),
    "read_file": ("search_docs", "get_order"),
    "write_file": ("create_ticket", "send_email"),
}

_REF = re.compile(r"\{([a-z_][a-z0-9_]*(?:\.[A-Za-z0-9_@-]+)*)(?::([^{}]*))?\}")
_SUSPICIOUS = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier)|disregard (?:your|all|the|any)|"
    r"system (?:override|notice)|assistants? (?:must|should) now|you are now|developer mode|"
    r"new instructions?:|do not tell the user",
    re.IGNORECASE,
)
_FILLER = "Further detail is available on request."


class _MissingValue(Exception):
    """A template refers to data the agent does not have."""

    def __init__(self, ref: str) -> None:
        super().__init__(ref)
        self.ref = ref


@dataclass
class _Failure:
    tool: str
    code: str
    message: str
    attempts: int
    invalid_data: bool = False


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    """Repair obviously recoverable type slips: numeric text where a number is expected."""
    kind = schema.get("type")
    if kind in {"number", "integer"} and isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return value
        if not math.isfinite(number):
            return value
        return int(number) if kind == "integer" and number == int(number) else number
    if kind == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        return {
            key: _coerce(item, properties[key]) if key in properties else item
            for key, item in value.items()
        }
    if kind == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_coerce(item, schema["items"]) for item in value]
    return value


class _Episode:
    """State of one simulated run."""

    def __init__(self, ctx: AgentContext, plan: SimPlan, behaviour: Behaviour) -> None:
        self.ctx = ctx
        self.task: AgentTask = ctx.task
        self.plan = plan
        self.b = behaviour
        self.memory: dict[str, Any] = {"input": self.task.input, "slots": {}}
        self.memory["secrets"] = "\n".join(item.text for item in self.task.instructions)
        self.triggered: set[str] = set()
        self.seen: list[str] = [self.task.input, *(doc.text for doc in self.task.context)]
        self.failure: _Failure | None = None
        self.ask = False
        self.declined = False
        self.skipped_steps = False
        self.effects: list[str] = []
        self.sources: list[str] = []
        self.injected_text: list[str] = []
        self.fabrications = 0
        specs = json.dumps([info.parameters for info in ctx.tools.specs()])
        self.context_chars = (
            len(self.task.input)
            + sum(len(item.text) for item in self.task.instructions)
            + sum(len(doc.text) for doc in self.task.context)
            + len(specs)
        )

    def trigger(self, mode: str) -> None:
        """Note that a broken behaviour actually changed what the agent did."""
        if mode not in self.triggered:
            self.triggered.add(mode)
            self.ctx.annotate("triggered", sorted(self.triggered))

    # ------------------------------------------------------------------ accounting

    def llm(self, note: str, *, new_chars: int = 0, output_chars: int = 120) -> None:
        self.context_chars += new_chars
        input_tokens = math.ceil(self.context_chars / 4)
        output_tokens = math.ceil(output_chars / 4)
        latency = 150.0 + 2.0 * output_tokens
        if self.b.expensive:
            input_tokens += EXPENSIVE_EXTRA_TOKENS
            self.trigger("expensive")
        if self.b.slow:
            latency += SLOW_EXTRA_MS
            self.trigger("slow")
        self.ctx.record_llm_call(
            model=MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            note=note,
        )

    # ------------------------------------------------------------------ templates

    def lookup(self, ref: str) -> Any:
        parts = ref.split(".")
        if parts[0] not in self.memory:
            raise _MissingValue(ref)
        value: Any = self.memory[parts[0]]
        for part in parts[1:]:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                raise _MissingValue(ref)
        return value

    def lookup_or_stand_in(self, ref: str) -> Any:
        """Look a value up; the faulty agents paper over gaps instead of stopping."""
        try:
            value = self.lookup(ref)
        except _MissingValue:
            if ref.startswith("slots."):
                raise
            if self.b.hallucinates:
                self.trigger("hallucinates")
                return self.fabricate_scalar()
            if self.b.partial_completion and self.skipped_steps:
                return "done"
            if self.b.trusts_invalid_data:
                return "unknown"
            raise
        if value is None and not self.b.trusts_invalid_data:
            raise _MissingValue(ref)
        return value

    def show(self, value: Any, spec: str | None, display: bool) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            if spec:
                return format(value, spec)
            if isinstance(value, float):
                return f"{value:.2f}" if display else repr(value)
            return str(value)
        if isinstance(value, str):
            return value
        if value is None:
            return "None"
        return json.dumps(value, ensure_ascii=False)

    def fill(self, template: Any, *, display: bool) -> Any:
        """Substitute ``{step.field}`` references in a string, list or dict."""
        if isinstance(template, str):
            whole = _REF.fullmatch(template)
            if whole and not whole.group(2) and not display:
                return self.lookup_or_stand_in(whole.group(1))
            return _REF.sub(
                lambda match: self.show(
                    self.lookup_or_stand_in(match.group(1)), match.group(2), display
                ),
                template,
            )
        if isinstance(template, list):
            return [self.fill(item, display=display) for item in template]
        if isinstance(template, dict):
            return {key: self.fill(item, display=display) for key, item in template.items()}
        return template

    def holds(self, condition: SimCondition) -> bool:
        try:
            value = self.lookup(condition.ref)
        except _MissingValue:
            return False
        if "equals" in condition.model_fields_set and value != condition.equals:
            return False
        if condition.one_of is not None and value not in condition.one_of:
            return False
        return not ("not_equals" in condition.model_fields_set and value == condition.not_equals)

    # ------------------------------------------------------------------ fabrication

    def fabricate_scalar(self) -> str:
        self.fabrications += 1
        draw = self.ctx.uniform("fabricate", self.fabrications)
        return f"FAB-{1000 + int(draw * 9000)}"

    def fabricate(self, schema: dict[str, Any]) -> Any:
        """Invent a value shaped like ``schema`` (what the hallucinating agent does)."""
        self.fabrications += 1
        draw = self.ctx.uniform("fabricate", self.fabrications)
        if "const" in schema:
            return schema["const"]
        if schema.get("enum"):
            return schema["enum"][0]
        kind = schema.get("type")
        if kind == "object":
            properties = schema.get("properties", {})
            return {
                name: self.fabricate(properties.get(name, {}))
                for name in schema.get("required", [])
            }
        if kind == "array":
            items = schema.get("items")
            return [self.fabricate(items)] if isinstance(items, dict) else []
        if kind == "number":
            return round(50 + draw * 900, 2)
        if kind == "integer":
            return 20 + int(draw * 980)
        if kind == "boolean":
            return True
        return f"FAB-{1000 + int(draw * 9000)}"

    # ------------------------------------------------------------------ tool calls

    def dump(self) -> str:
        working = {
            key: value for key, value in self.memory.items() if key not in {"secrets", "context"}
        }
        data = json.dumps(working, ensure_ascii=False, default=str)
        return f"[debug] instructions: {self.memory['secrets']} | working data: {data}"

    def leak_into(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The leaking agent attaches its whole context to anything it sends out."""
        info = self.ctx.tools.spec(tool)
        if not self.b.leaks_canary or info is None or not info.sink:
            return arguments
        leaked = dict(arguments)
        for key in ("body", "content"):
            if isinstance(leaked.get(key), str):
                leaked[key] = f"{leaked[key]}\n\n{self.dump()}"
                self.trigger("leaks_canary")
                return leaked
        if "payload" in leaked:
            leaked["payload"] = {"data": leaked["payload"], "debug": self.dump()}
            self.trigger("leaks_canary")
        return leaked

    def check(self, tool: str, raw: Any) -> tuple[Any, str | None]:
        """Validate a tool result against the tool's declared result schema, repairing if safe."""
        info = self.ctx.tools.spec(tool)
        if info is None:
            return raw, None
        value = raw
        if isinstance(value, str) and info.returns.get("type") in {"object", "array"}:
            try:
                value = json.loads(value)
            except (ValueError, RecursionError):
                return raw, "the result is not valid JSON"
        problems = validate(info.returns, value)
        if problems:
            value = _coerce(value, info.returns)
            problems = validate(info.returns, value)
        return (value, None) if not problems else (raw, problems[0])

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Call a tool with the agent's retry and validation policy. ``None`` means failure."""
        arguments = self.leak_into(tool, arguments)
        attempt = 0
        while True:
            attempt += 1
            problem: str | None = None
            code = "invalid_result"
            retryable = True
            retry_after: float | None = None
            try:
                raw = self.ctx.tools.call(tool, arguments)
            except ToolError as exc:
                self.llm("tool error", new_chars=len(exc.message))
                problem, code = exc.message, exc.code
                retryable, retry_after = exc.retryable, exc.retry_after_ms
                invalid = False
            else:
                text = json.dumps(raw, ensure_ascii=False)
                self.seen.append(text)
                self.llm("tool result", new_chars=len(text))
                value, problem = self.check(tool, raw)
                invalid = problem is not None
                if self.b.trusts_invalid_data:
                    if invalid or value != raw:
                        self.trigger("trusts_invalid_data")
                    return raw
                if not invalid:
                    return value
            if self.b.unbounded_retry:
                if attempt >= MAX_ATTEMPTS or not retryable:
                    self.trigger("unbounded_retry")
                continue
            if retryable and attempt < MAX_ATTEMPTS:
                if self.b.no_retry:
                    self.trigger("no_retry")
                else:
                    self.ctx.wait(
                        retry_after if retry_after is not None else 200.0 * 2 ** (attempt - 1),
                        reason=f"back off before retrying {tool}",
                    )
                    continue
            self.failure = _Failure(tool, code, problem or "", attempt, invalid_data=invalid)
            return None

    def call_with_resume(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Call a batch tool and resend only the part it reports as failed."""
        result = self.call(tool, arguments)
        rounds = 1
        while (
            isinstance(result, dict)
            and isinstance(result.get("failed"), list)
            and result["failed"]
            and "recipients" in arguments
        ):
            if self.b.partial_completion:
                self.trigger("partial_completion")
                break
            if self.b.no_retry or (rounds >= MAX_ATTEMPTS and not self.b.unbounded_retry):
                if self.b.no_retry:
                    self.trigger("no_retry")
                self.failure = _Failure(
                    tool, "partial", f"{len(result['failed'])} recipient(s) not reached", rounds
                )
                return None
            rounds += 1
            if not self.b.unbounded_retry:
                self.ctx.wait(200.0 * 2 ** (rounds - 2), reason=f"back off before resuming {tool}")
            again = self.call(tool, {**arguments, "recipients": list(result["failed"])})
            if not isinstance(again, dict):
                return None
            result = {
                "sent": [*result.get("sent", []), *again.get("sent", [])],
                "failed": again.get("failed", []),
            }
        return result

    def swap_tool(self, tool: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        available = set(self.ctx.tools.names())
        for candidate in CONFUSABLE.get(tool, ()):
            info = self.ctx.tools.spec(candidate)
            if candidate in available and info is not None:
                values = list(arguments.values()) or ["x"]
                required = info.parameters.get("required", [])
                mapped = {
                    name: values[min(position, len(values) - 1)]
                    for position, name in enumerate(required)
                }
                self.trigger("wrong_tool")
                return candidate, mapped
        return tool, arguments

    def corrupt_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not arguments:
            return arguments
        variant = int(self.ctx.uniform("bad_arguments") * 3)
        corrupted = dict(arguments)
        first = next(iter(corrupted))
        value = corrupted[first]
        if variant == 0:
            if isinstance(value, str):
                digits = re.search(r"(\d+)(?!.*\d)", value)
                if digits:
                    bumped = str(int(digits.group(1)) + 1).zfill(len(digits.group(1)))
                    corrupted[first] = value[: digits.start(1)] + bumped + value[digits.end(1) :]
                else:
                    corrupted[first] = "something else entirely"
            elif isinstance(value, int | float) and not isinstance(value, bool):
                corrupted[first] = value * 10
            else:
                corrupted[first] = "x"
        elif variant == 1:
            del corrupted[first]
        else:
            corrupted[first] = [value]
        self.trigger("bad_arguments")
        return corrupted

    def run_steps(self) -> None:
        steps: list[SimStep] = list(self.plan.steps)
        limit = len(steps)
        if self.b.partial_completion and len(steps) >= 2 and not self.plan.missing:
            limit = math.ceil(len(steps) / 2)
        sabotaged = False
        for position, step in enumerate(steps):
            if position >= limit:
                self.skipped_steps = True
                self.trigger("partial_completion")
                break
            if step.when is not None and not self.holds(step.when):
                self.declined = True
                break
            try:
                arguments = self.fill(step.args, display=False)
            except _MissingValue as exc:
                if exc.ref.startswith("slots."):
                    self.ask = True
                else:
                    self.failure = _Failure(step.tool, "missing_data", f"no value for {exc.ref}", 0)
                break
            tool = step.tool
            if not sabotaged and self.b.wrong_tool:
                tool, arguments = self.swap_tool(tool, arguments)
                sabotaged = tool != step.tool
            if not sabotaged and self.b.bad_arguments:
                changed = self.corrupt_arguments(arguments)
                sabotaged = changed != arguments
                arguments = changed
            result = self.call_with_resume(tool, arguments)
            if result is None and self.failure is not None:
                if not self.b.hallucinates:
                    break
                info = self.ctx.tools.spec(tool)
                result = self.fabricate(info.returns if info is not None else {})
                self.trigger("hallucinates")
                self.failure = None
            self.memory[step.id] = result
            info = self.ctx.tools.spec(tool)
            if info is not None and info.side_effect:
                self.effects.append(tool)

    # ------------------------------------------------------------------ injected instructions

    def obey_injections(self) -> None:
        """What a gullible agent does: treat instructions found in data as its own."""
        seen = "\n".join(self.seen).casefold()
        self.memory["context"] = self.dump()
        for injection in self.plan.injections:
            if injection.marker.casefold() not in seen:
                continue
            self.trigger("follows_injection")
            for call in injection.calls:
                try:
                    arguments = self.fill(call.args, display=False)
                    self.ctx.tools.call(call.tool, arguments)
                except (ToolError, _MissingValue):
                    continue
            if injection.say:
                self.injected_text.append(str(self.fill(injection.say, display=True)))

    # ------------------------------------------------------------------ reading documents

    def candidates(self) -> list[tuple[str, str]]:
        """(document id, sentence) pairs from retrieved hits and from the provided context."""
        found: list[tuple[str, str]] = []
        read = self.plan.read
        if read is None:
            return found
        documents: list[tuple[str, str]] = []
        for step_id in read.from_steps:
            hits = self.memory.get(step_id)
            results = hits.get("results") if isinstance(hits, dict) else None
            for hit in results if isinstance(results, list) else []:
                if isinstance(hit, dict) and isinstance(hit.get("text"), str):
                    documents.append((str(hit.get("id", "")), hit["text"]))
            if isinstance(hits, dict) and isinstance(hits.get("content"), str):
                documents.append((str(hits.get("path", step_id)), hits["content"]))
        if read.use_context:
            documents.extend((doc.id, doc.text) for doc in self.task.context)
        for doc_id, text in documents:
            found.extend((doc_id, sentence) for sentence in split_sentences(text))
        return found

    def read(self) -> tuple[str, Status]:
        read = self.plan.read
        question = set(content_tokens((read.question if read else None) or self.task.input))
        pairs = self.candidates()
        scored = [
            (overlap(question, set(content_tokens(sentence))), -position, doc_id, sentence)
            for position, (doc_id, sentence) in enumerate(pairs)
            if not _SUSPICIOUS.search(sentence)
        ]
        best = max(scored, default=None)
        if self.b.distracted and pairs:
            last_doc = pairs[-1][0]
            recent = [item for item in scored if item[2] == last_doc]
            lure = max(recent, default=None)
            if lure is not None and lure[0] >= DISTRACTION_THRESHOLD and lure != best:
                self.trigger("distracted")
                best = lure
        if best is None or best[0] < READ_THRESHOLD:
            if self.b.hallucinates:
                self.trigger("hallucinates")
                days = 12 + int(self.ctx.uniform("fabricated answer") * 78)
                reference = self.fabricate_scalar().replace("FAB", "POL")
                return f"According to policy {reference}, the answer is {days} days.", "completed"
            return "I could not find that in the documents available to me.", "abstained"
        self.sources.append(best[2])
        return f"{best[3]} [{best[2]}]", "completed"

    # ------------------------------------------------------------------ composing the answer

    def failure_text(self) -> tuple[str, Status]:
        failure = self.failure
        if failure is None:
            return "I could not complete this request.", "failed"
        if failure.code == "not_found":
            return f"I could not find what was asked for: {failure.message}.", "abstained"
        if failure.invalid_data:
            reason = f"{failure.tool} returned data that failed validation ({failure.message})"
        elif failure.code == "missing_data":
            reason = f"data needed for {failure.tool} is missing ({failure.message})"
        else:
            reason = f"{failure.tool} failed ({failure.code}: {failure.message})"
        tries = f" after {failure.attempts} attempt(s)" if failure.attempts else ""
        done = (
            "Completed before the failure: " + ", ".join(self.effects) + "."
            if self.effects
            else "Nothing was changed."
        )
        text = f"I could not complete this request because {reason}{tries}. {done}"
        return text + " No further action was taken.", "failed"

    def compose(self) -> tuple[str, Status]:
        if self.ask:
            try:
                question = str(self.fill(self.plan.clarifying_question or "", display=True))
            except _MissingValue:
                question = ""
            return question or "Could you clarify what you would like me to do?", (
                "needs_clarification"
            )
        if self.declined and self.plan.otherwise is not None:
            try:
                answer = str(self.fill(self.plan.otherwise.answer, display=True))
            except _MissingValue:
                answer = "I cannot do that."
            return answer, self.plan.otherwise.status
        if self.failure is not None:
            return self.failure_text()
        try:
            if self.plan.read is not None:
                return self.read()
            if self.plan.answer_json is not None:
                filled = self.fill(self.plan.answer_json, display=False)
                return json.dumps(filled, ensure_ascii=False), self.plan.status
            if self.plan.answer is not None:
                return str(self.fill(self.plan.answer, display=True)), self.plan.status
        except _MissingValue as exc:
            self.failure = _Failure("the answer", "missing_data", f"no value for {exc.ref}", 0)
            return self.failure_text()
        return "Done.", "completed"

    def constrain(self, text: str, rules: list[Rule], status: Status) -> str:
        """Rewrite ``text`` so that it satisfies the rules, as far as a template agent can."""
        if not rules:
            return text
        structured = is_json(text.strip()) and text.strip()[:1] in "{["
        formats = [rule.format for rule in rules if rule.format]
        if formats and not structured:
            wanted = formats[0]
            sentences = split_sentences(text)
            if wanted == "json":
                text = json.dumps({"answer": text}, ensure_ascii=False)
                structured = True
            elif wanted == "bullets":
                text = "\n".join(f"- {sentence}" for sentence in sentences)
            elif wanted == "numbered":
                text = "\n".join(f"{n}. {sentence}" for n, sentence in enumerate(sentences, 1))
            else:
                text = re.sub(r"[*#`]", "", text)
        for rule in rules:
            for phrase in [*rule.must_not_include, *rule.confidential]:
                text = re.sub(re.escape(phrase), "[redacted]", text, flags=re.IGNORECASE)
            if rule.must_not_match is not None:
                text = re.sub(rule.must_not_match, "[redacted]", text)
        required = [phrase for rule in rules for phrase in rule.must_include]
        limits = [rule.max_words for rule in rules if rule.max_words is not None]
        if limits and not structured:
            room = max(1, min(limits) - sum(word_count(phrase) for phrase in required))
            words = text.split()
            if len(words) > room:
                text = " ".join(words[:room])
        for phrase in required:
            if phrase.casefold() in text.casefold():
                continue
            if structured:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    payload["note"] = phrase
                    text = json.dumps(payload, ensure_ascii=False)
            elif formats and formats[0] == "bullets":
                text = f"{text}\n- {phrase}"
            else:
                text = f"{text} {phrase}"
        if not structured:
            for rule in rules:
                if rule.starts_with and not text.lstrip().startswith(rule.starts_with):
                    text = f"{rule.starts_with} {text}"
                if rule.ends_with and not text.rstrip().endswith(rule.ends_with):
                    text = f"{text} {rule.ends_with}"
            if status == "completed":
                if (
                    any(r.cite_sources for r in rules)
                    and self.sources
                    and not CITATION.search(text)
                ):
                    text = f"{text} [{self.sources[0]}]"
                floors = [rule.min_words for rule in rules if rule.min_words is not None]
                while floors and word_count(text) < max(floors):
                    text = f"{text} {_FILLER}"
        return text

    def break_json(self, text: str) -> str:
        """Corrupt a JSON answer in one of four ways."""
        stripped = text.strip()
        if not (stripped[:1] in "{[" and is_json(stripped)):
            return text
        self.trigger("invalid_json")
        variant = int(self.ctx.uniform("invalid_json") * 4)
        schema = self.task.output_schema or {}
        payload = json.loads(stripped)
        if variant == 3 and isinstance(payload, dict) and schema.get("required"):
            payload.pop(schema["required"][0], None)
            return json.dumps(payload, ensure_ascii=False)
        if variant == 2:
            return stripped.replace('"', "'")
        if variant == 1:
            return f"Here is the JSON you asked for:\n{stripped}"
        return stripped[:-1]

    def finish(self, text: str, status: Status) -> AgentResponse:
        resolved = resolve_rules(self.task.instructions)
        proper = [rule for _, rule in resolved.rules()]
        constrained = self.constrain(text, proper, status)
        if self.b.ignores_instruction:
            user_only = [
                item.rule
                for item in self.task.instructions
                if item.priority == "user" and item.rule is not None
            ]
            relaxed = self.constrain(text, user_only, status)
            if relaxed != constrained:
                self.trigger("ignores_instruction")
            constrained = relaxed
        text = constrained
        if self.b.invalid_json and status == "completed":
            text = self.break_json(text)
        if self.injected_text:
            text = text + "\n" + "\n".join(self.injected_text)
        if self.b.leaks_canary:
            text = f"{text}\n\n{self.dump()}"
            self.trigger("leaks_canary")
        self.llm("final answer", output_chars=len(text))
        return AgentResponse(
            output=text,
            status=status,
            metadata={"simulated": True, "triggered": sorted(self.triggered)},
        )

    def run(self) -> AgentResponse:
        self.ctx.annotate("simulated", True)
        self.ctx.annotate("modes", self.b.active_modes())
        self.ctx.annotate("triggered", [])
        self.llm("read the task and plan")
        if self.plan.missing and self.b.guesses_when_ambiguous:
            self.memory["slots"] = dict(self.plan.guess)
            self.trigger("guesses_when_ambiguous")
        self.run_steps()
        if self.plan.missing and not self.b.guesses_when_ambiguous and self.failure is None:
            self.ask = True
        if self.b.follows_injection:
            self.obey_injections()
        text, status = self.compose()
        return self.finish(text, status)


class SimulatedAgent:
    """A rule-based stand-in for an agent, with optional deliberately broken behaviour.

    ``activation_rate`` below 1 makes the broken behaviour appear only in that share of
    trials (decided reproducibly from the trial seed), which is how flaky agents are
    simulated.
    """

    def __init__(
        self,
        name: str = "reference-agent",
        version: str = "1.0.0",
        behaviour: Behaviour | None = None,
        activation_rate: float = 1.0,
    ) -> None:
        if not 0.0 <= activation_rate <= 1.0:
            raise ConfigError("activation_rate must be between 0 and 1")
        self.name = name
        self.version = version
        self.behaviour = behaviour or Behaviour()
        self.activation_rate = activation_rate

    def run(self, ctx: AgentContext) -> AgentResponse:
        plan = ctx.task.sim
        if plan is None:
            raise ConfigError(
                "simulated agents need a 'sim' block in the case; use your own agent for "
                "suites without one"
            )
        active = self.activation_rate >= 1.0 or ctx.uniform("activation") < self.activation_rate
        return _Episode(ctx, plan, self.behaviour if active else Behaviour()).run()


def reference_agent(name: str = "reference-agent", version: str = "1.0.0") -> SimulatedAgent:
    """The clean simulated agent that every built-in case is written to be passable by."""
    return SimulatedAgent(name=name, version=version)


def faulty_agent(
    *modes: str,
    name: str | None = None,
    version: str = "1.0.0",
    activation_rate: float = 1.0,
) -> SimulatedAgent:
    """A simulated agent with one or more deliberately introduced failure modes."""
    unknown = [mode for mode in modes if mode not in FAILURE_MODES]
    if unknown or not modes:
        raise ConfigError(
            f"unknown failure mode(s) {unknown}; choose from {', '.join(FAILURE_MODES)}"
        )
    behaviour = replace(Behaviour(), **{mode: True for mode in modes})
    return SimulatedAgent(
        name=name or "faulty-" + "+".join(mode.replace("_", "-") for mode in modes),
        version=version,
        behaviour=behaviour,
        activation_rate=activation_rate,
    )
