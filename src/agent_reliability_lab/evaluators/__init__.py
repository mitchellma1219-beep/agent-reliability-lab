"""Deterministic evaluators.

Each evaluator is a pure function of a case and a recorded trace. ``DEFAULT_EVALUATORS``
lists them in the order they are applied; an evaluator that has nothing to check for a case
is skipped rather than reported as passed.
"""

from __future__ import annotations

from .base import EvalContext, Evaluator
from .budget import CostBudget, LatencyBudget
from .groundedness import Groundedness
from .hallucination import Hallucination
from .instruction_following import InstructionFollowing
from .output_schema import OutputSchema
from .recovery import Recovery
from .security import CanaryLeak, Containment, InjectionResistance
from .task_success import TaskSuccess
from .tool_calls import ToolCallAccuracy
from .workflow import Workflow

DEFAULT_EVALUATORS: tuple[Evaluator, ...] = (
    TaskSuccess(),
    ToolCallAccuracy(),
    InstructionFollowing(),
    Groundedness(),
    Hallucination(),
    OutputSchema(),
    Workflow(),
    InjectionResistance(),
    CanaryLeak(),
    Recovery(),
    LatencyBudget(),
    CostBudget(),
    Containment(),
)

EVALUATOR_NAMES: tuple[str, ...] = tuple(item.name for item in DEFAULT_EVALUATORS)

__all__ = [
    "DEFAULT_EVALUATORS",
    "EVALUATOR_NAMES",
    "CanaryLeak",
    "Containment",
    "CostBudget",
    "EvalContext",
    "Evaluator",
    "Groundedness",
    "Hallucination",
    "InjectionResistance",
    "InstructionFollowing",
    "LatencyBudget",
    "OutputSchema",
    "Recovery",
    "TaskSuccess",
    "ToolCallAccuracy",
    "Workflow",
]
