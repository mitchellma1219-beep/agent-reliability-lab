"""The contract between the lab and an agent under test.

An agent is anything with a ``name``, a ``version`` and a ``run(ctx)`` method. It receives an
:class:`AgentContext` holding the task (never the expected answer), the simulated tools and
hooks for reporting model usage, and it returns its final answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, JsonValue

from ..faults import uniform
from ..models import AgentTask, Model, Status
from ..sandbox import Sandbox
from ..tools import ToolInfo


class AgentResponse(Model):
    """Final answer of an agent.

    ``status`` lets an agent say explicitly what kind of answer this is (a clarifying
    question, a refusal, an explicit failure). If it is left out, the lab infers the status
    from the wording, which is less reliable.
    """

    output: str
    status: Status | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ToolBox:
    """The tools an agent may call. Every call goes through the sandbox and is recorded."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.__sandbox = sandbox

    def names(self) -> list[str]:
        return [info.name for info in self.__sandbox.infos()]

    def specs(self) -> list[ToolInfo]:
        return self.__sandbox.infos()

    def spec(self, name: str) -> ToolInfo | None:
        found = self.__sandbox.spec(name)
        return None if found is None else found.info()

    def call(self, name: str, arguments: Mapping[str, Any] | None = None, /, **kwargs: Any) -> Any:
        """Call a tool with a dict of arguments, keyword arguments, or both."""
        if kwargs:
            merged: dict[str, Any] = dict(arguments or {})
            merged.update(kwargs)
            return self.__sandbox.call(name, merged)
        return self.__sandbox.call(name, arguments)


class AgentContext:
    """Everything an agent can see and do while handling one case."""

    def __init__(self, task: AgentTask, sandbox: Sandbox, seed: int) -> None:
        self.task = task
        self.tools = ToolBox(sandbox)
        self.seed = seed
        self.__sandbox = sandbox

    def record_llm_call(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        model: str = "unknown",
        note: str = "",
    ) -> None:
        """Report one model call so that tokens, cost and latency can be accounted for."""
        self.__sandbox.record_llm_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            note=note,
        )

    def wait(self, ms: float, reason: str = "") -> None:
        """Pause for ``ms`` of simulated time (back-off, Retry-After)."""
        self.__sandbox.wait(ms, reason)

    def annotate(self, key: str, value: Any) -> None:
        """Attach JSON-like metadata to the trace. Unlike response metadata it survives an abort."""
        self.__sandbox.annotate(key, value)

    @property
    def elapsed_ms(self) -> float:
        """Accounted time so far: tool latency, reported model latency and waits."""
        return self.__sandbox.now_ms

    def uniform(self, *key: object) -> float:
        """A reproducible number in [0, 1) for this case, trial and key."""
        return uniform(self.seed, *key)


@runtime_checkable
class Agent(Protocol):
    """Structural type of an agent under test."""

    name: str
    version: str

    def run(self, ctx: AgentContext) -> AgentResponse | str: ...


# --------------------------------------------------------------------------------------
# Status inference for agents that return plain text
# --------------------------------------------------------------------------------------

_FAILED = re.compile(
    r"\b(?:unable to (?:complete|finish|process|retrieve|reach|access)|could ?n[o']t (?:complete|"
    r"finish|process|retrieve|reach|access|be completed)|failed to|was not able to|"
    r"temporarily unavailable|is unavailable|is (?:currently )?down|try again later|"
    r"an error occurred|did not succeed)\b",
    re.IGNORECASE,
)
_REFUSED = re.compile(
    r"\b(?:i (?:can(?:no|')t|won't|will not|am not able to|am unable to) (?:help|share|provide|"
    r"do that|comply|assist|reveal|disclose|issue|process)|not (?:allowed|permitted|authori[sz]ed)"
    r" to|against (?:the |our |my )?polic)",
    re.IGNORECASE,
)
_ABSTAINED = re.compile(
    r"\b(?:(?:could ?n[o']t|cannot|can't|did not|didn't|unable to) (?:find|locate|determine)|"
    r"(?:do not|don't|does not|doesn't) (?:have|contain|include|mention|say|state) (?:enough |"
    r"any |that |this |the )?(?:information|details?|data|answer)?|no (?:information|record|"
    r"matching|such)|not (?:available|mentioned|stated|specified|covered) in)",
    re.IGNORECASE,
)
_CLARIFY = re.compile(
    r"\b(?:which (?:one|order|customer|product|account|of)|could you (?:please )?(?:clarify|"
    r"confirm|specify|tell me|provide|share)|can you (?:please )?(?:clarify|confirm|specify|"
    r"tell me|provide)|do you mean|did you mean|please (?:clarify|confirm|specify|provide)|"
    r"what (?:is|was) the (?:order|customer|account|product))\b",
    re.IGNORECASE,
)


def infer_status(output: str) -> Status:
    """Guess the kind of answer from its wording.

    A heuristic for agents that return plain text. Explicit failure wins over refusal, then
    abstention, then a clarifying question; anything else counts as ``completed``. See
    ``docs/evaluator-limitations.md`` for the ways this goes wrong.
    """
    if _FAILED.search(output):
        return "failed"
    if _REFUSED.search(output):
        return "refused"
    if _ABSTAINED.search(output):
        return "abstained"
    if "?" in output and _CLARIFY.search(output):
        return "needs_clarification"
    return "completed"
