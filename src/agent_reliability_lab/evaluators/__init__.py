"""Deterministic evaluators."""

from __future__ import annotations

from .base import EvalContext, Evaluator
from .task_success import TaskSuccess

DEFAULT_EVALUATORS: tuple[Evaluator, ...] = (TaskSuccess(),)

EVALUATOR_NAMES: tuple[str, ...] = tuple(item.name for item in DEFAULT_EVALUATORS)

__all__ = ["DEFAULT_EVALUATORS", "EVALUATOR_NAMES", "EvalContext", "Evaluator", "TaskSuccess"]
