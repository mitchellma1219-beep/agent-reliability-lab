"""Adapters that turn ordinary Python objects into agents the lab can run.

* :class:`FunctionAgent` wraps a plain function ``(ctx) -> str``.
* :class:`ScriptedAgent` replays a fixed list of tool calls and a fixed answer. It is useful
  for testing evaluators and for replaying a recorded session.
* :class:`ToolLoopAgent` runs the usual "model asks for tools until it answers" loop around
  any chat model you supply as a callable. The lab never ships or calls a model itself.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import ToolError
from ..models import Status
from ..tools import ToolInfo
from .base import AgentContext, AgentResponse


class FunctionAgent:
    """An agent defined by a function that takes the context and returns the answer."""

    def __init__(
        self,
        function: Callable[[AgentContext], AgentResponse | str],
        name: str = "function-agent",
        version: str = "0",
    ) -> None:
        self._function = function
        self.name = name
        self.version = version

    def run(self, ctx: AgentContext) -> AgentResponse | str:
        return self._function(ctx)


@dataclass(frozen=True)
class ScriptedCall:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


class ScriptedAgent:
    """Makes the given tool calls in order (ignoring tool errors), then returns the answer."""

    def __init__(
        self,
        calls: Sequence[ScriptedCall | tuple[str, Mapping[str, Any]]] = (),
        output: str = "",
        status: Status | None = None,
        name: str = "scripted-agent",
        version: str = "0",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        self._calls = [
            call if isinstance(call, ScriptedCall) else ScriptedCall(call[0], call[1])
            for call in calls
        ]
        self._output = output
        self._status = status
        self._usage = (input_tokens, output_tokens, latency_ms)
        self.name = name
        self.version = version

    def run(self, ctx: AgentContext) -> AgentResponse:
        for call in self._calls:
            try:
                ctx.tools.call(call.tool, dict(call.arguments))
            except ToolError:
                continue
        input_tokens, output_tokens, latency_ms = self._usage
        if input_tokens or output_tokens or latency_ms:
            ctx.record_llm_call(
                model="scripted",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
        return AgentResponse(output=self._output, status=self._status)


@dataclass(frozen=True)
class ToolRequest:
    """A tool call requested by the model."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelTurn:
    """One reply of a chat model: text, tool requests, or both, plus reported usage."""

    text: str = ""
    tool_requests: Sequence[ToolRequest] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    status: Status | None = None


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class ChatModel(Protocol):
    """Anything that can continue a conversation given the available tools."""

    def __call__(self, messages: Sequence[Message], tools: Sequence[ToolInfo]) -> ModelTurn: ...


class ToolLoopAgent:
    """A minimal tool-calling loop around a caller-supplied chat model.

    The model callable receives the conversation so far and the tool descriptions, and
    returns a :class:`ModelTurn`. Tool errors are fed back to the model as text so that it can
    react to them. Token counts come from the model's own report; latency is measured.
    """

    def __init__(
        self,
        model: ChatModel,
        name: str = "tool-loop-agent",
        version: str = "0",
        model_name: str = "unknown",
        max_turns: int = 12,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._max_turns = max_turns
        self.name = name
        self.version = version

    @staticmethod
    def _opening(ctx: AgentContext) -> list[Message]:
        task = ctx.task
        messages: list[Message] = []
        if task.instructions:
            system = "\n".join(item.text for item in task.instructions if item.priority == "system")
            if system:
                messages.append(Message("system", system))
        parts = [task.input]
        parts.extend(item.text for item in task.instructions if item.priority == "user")
        for doc in task.context:
            parts.append(f"[document {doc.id}] {doc.title}\n{doc.text}")
        if task.output_schema is not None:
            parts.append("Answer with JSON matching this schema: " + json.dumps(task.output_schema))
        messages.append(Message("user", "\n\n".join(parts)))
        return messages

    def run(self, ctx: AgentContext) -> AgentResponse:
        messages = self._opening(ctx)
        tools = ctx.tools.specs()
        for _ in range(self._max_turns):
            started = time.perf_counter()
            turn = self._model(messages, tools)
            ctx.record_llm_call(
                model=self._model_name,
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            if not turn.tool_requests:
                return AgentResponse(output=turn.text, status=turn.status)
            if turn.text:
                messages.append(Message("assistant", turn.text))
            for request in turn.tool_requests:
                try:
                    result = ctx.tools.call(request.name, dict(request.arguments))
                    content = json.dumps(result, ensure_ascii=False)
                except ToolError as exc:
                    content = json.dumps(
                        {
                            "error": exc.code,
                            "message": exc.message,
                            "retryable": exc.retryable,
                            "retry_after_ms": exc.retry_after_ms,
                        }
                    )
                messages.append(Message("tool", f"{request.name}: {content}"))
        return AgentResponse(
            output="I could not complete this request within the allowed number of steps.",
            status="failed",
        )
