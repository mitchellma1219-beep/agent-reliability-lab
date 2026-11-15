"""Optional rubric judges (LLM-as-judge), calibration and human-review routing.

Nothing here runs unless it is asked for. The deterministic evaluators in
:mod:`agent_reliability_lab.evaluators` do not use a judge.
"""

from __future__ import annotations

from .base import Judge, decide
from .calibration import (
    CalibrationReport,
    CalibrationSet,
    builtin_calibration_set,
    calibrate,
    load_calibration_set,
    parse_calibration_set,
)
from .mock import MockJudge, ScriptedJudge, token_f1
from .models import Criterion, Decision, JudgeItem, JudgeVerdict, ReviewPolicy, Rubric

__all__ = [
    "CalibrationReport",
    "CalibrationSet",
    "Criterion",
    "Decision",
    "Judge",
    "JudgeItem",
    "JudgeVerdict",
    "MockJudge",
    "ReviewPolicy",
    "Rubric",
    "ScriptedJudge",
    "builtin_calibration_set",
    "calibrate",
    "decide",
    "load_calibration_set",
    "parse_calibration_set",
    "token_f1",
]
