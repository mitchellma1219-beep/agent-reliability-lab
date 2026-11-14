"""Agent Reliability Lab: test whether an AI agent is reliable, safe and affordable.

The public API is small:

>>> from agent_reliability_lab import load_builtin_suite, reference_agent, run_suite
>>> run = run_suite(load_builtin_suite("core"), reference_agent())
>>> all(result.passed for result in run.results)
True
"""

from __future__ import annotations

__version__ = "0.1.0"

from .agents import (
    Agent,
    AgentContext,
    AgentResponse,
    FunctionAgent,
    ScriptedAgent,
    ToolLoopAgent,
    faulty_agent,
    reference_agent,
)
from .errors import ArlError, ToolError
from .loader import load_suite, parse_suite
from .models import Case, RunConfig, RunResult, Suite
from .runner import run_case, run_suite
from .scenarios import builtin_suite_names, load_builtin_suite

__all__ = [
    "Agent",
    "AgentContext",
    "AgentResponse",
    "ArlError",
    "Case",
    "FunctionAgent",
    "RunConfig",
    "RunResult",
    "ScriptedAgent",
    "Suite",
    "ToolError",
    "ToolLoopAgent",
    "__version__",
    "builtin_suite_names",
    "faulty_agent",
    "load_builtin_suite",
    "load_suite",
    "parse_suite",
    "reference_agent",
    "run_case",
    "run_suite",
]
