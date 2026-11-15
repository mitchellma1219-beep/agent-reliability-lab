"""Models for rubric-based judging."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..models import ID_PATTERN, MAX_TEXT_CHARS, Model

Label = Literal["pass", "fail"]
FinalLabel = Literal["pass", "fail", "needs_human_review"]


class Criterion(Model):
    id: str = Field(pattern=ID_PATTERN)
    description: str = Field(min_length=1, max_length=2000)
    weight: float = Field(default=1.0, gt=0)


class Rubric(Model):
    """What a good answer looks like, broken into weighted criteria."""

    id: str = Field(pattern=ID_PATTERN)
    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(default="", max_length=4000)
    criteria: list[Criterion] = Field(min_length=1, max_length=20)
    pass_threshold: float = Field(default=0.7, gt=0.0, lt=1.0)


class JudgeItem(Model):
    """One answer to be judged, optionally with a label from a human for calibration."""

    id: str = Field(pattern=ID_PATTERN)
    task: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    output: str = Field(max_length=MAX_TEXT_CHARS)
    reference: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
    context: list[str] = Field(default_factory=list)
    human_label: Label | None = None
    note: str = Field(default="", max_length=1000)


class JudgeVerdict(Model):
    """One reading of one item by a judge."""

    item_id: str
    label: Label
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    criteria: dict[str, float] = Field(default_factory=dict)
    judge: str


class ReviewPolicy(Model):
    """When a verdict is not trusted and the item goes to a person instead."""

    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    ambiguity_margin: float = Field(default=0.1, ge=0.0, le=0.5)
    samples: int = Field(default=1, ge=1, le=9)


class Decision(Model):
    """The outcome for one item after the review policy has been applied."""

    item_id: str
    label: FinalLabel
    score: float | None = None
    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    verdicts: list[JudgeVerdict] = Field(default_factory=list)
