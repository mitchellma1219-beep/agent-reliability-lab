"""Calibrating a judge against labelled examples.

The judge and the review policy are run over items that already carry a label from a person
(or, for the bundled demonstration set, a label written by the fixture's author). The
report answers three questions: how often does the judge decide at all, how often is it
right when it decides, and in which direction is it wrong.

Convention: **positive means "pass"**. A *false positive* is a bad answer the judge let
through (false pass); a *false negative* is a good answer the judge rejected (false fail).
Agreement, kappa and the error rates are computed on the items the judge decided; items sent
to human review are counted separately and reported as the review rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from ..errors import JudgeError, SuiteError
from ..loader import SUFFIXES, parse_document
from ..metrics import Rate
from ..models import Model
from ..stats import cohens_kappa
from .base import Judge, decide
from .models import Decision, JudgeItem, ReviewPolicy, Rubric


class ConfusionMatrix(Model):
    """Rows are the human label, columns the judge's decision."""

    true_pass: int = 0
    false_pass: int = 0
    false_fail: int = 0
    true_fail: int = 0
    review_of_pass: int = 0
    review_of_fail: int = 0


class Disagreement(Model):
    item_id: str
    human: str
    judge: str
    score: float | None = None
    confidence: float | None = None
    note: str = ""


class CalibrationReport(Model):
    judge: str
    rubric: str
    dataset: str = ""
    policy: ReviewPolicy
    items: int
    labelled: int
    decided: int
    needs_review: int
    review_rate: float | None = None
    confusion: ConfusionMatrix
    agreement: Rate
    kappa: float | None = None
    false_positive_rate: Rate
    false_negative_rate: Rate
    disagreements: list[Disagreement] = Field(default_factory=list)
    review_reasons: dict[str, int] = Field(default_factory=dict)
    decisions: list[Decision] = Field(default_factory=list)


class CalibrationSet(Model):
    """A rubric plus labelled items, as stored in a calibration file."""

    rubric: Rubric
    items: list[JudgeItem] = Field(min_length=1, max_length=10_000)
    description: str = ""


def parse_calibration_set(text: str, fmt: str = "yaml", source: str = "<string>") -> CalibrationSet:
    data: Any = parse_document(text, fmt, source)
    try:
        loaded = CalibrationSet.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise JudgeError(f"{source}: invalid calibration set ({where}: {first['msg']})") from exc
    ids = [item.id for item in loaded.items]
    if len(ids) != len(set(ids)):
        raise JudgeError(f"{source}: item ids must be unique")
    return loaded


def load_calibration_set(path: str | Path) -> CalibrationSet:
    file = Path(path)
    fmt = SUFFIXES.get(file.suffix.lower())
    if fmt is None or not file.is_file():
        raise JudgeError(f"{file}: expected an existing .yaml, .yml or .json file")
    try:
        return parse_calibration_set(file.read_text(encoding="utf-8"), fmt, str(file))
    except SuiteError as exc:
        raise JudgeError(str(exc)) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise JudgeError(f"{file}: cannot be read as UTF-8 text") from exc


def builtin_calibration_set() -> CalibrationSet:
    """The bundled demonstration set. Its labels were written by the fixture's author."""
    from importlib import resources

    resource = resources.files("agent_reliability_lab.judge").joinpath("calibration_demo.yaml")
    return parse_calibration_set(resource.read_text(encoding="utf-8"), "yaml", "builtin")


def _reason_key(reason: str) -> str:
    if "disagreed with itself" in reason:
        return "self-disagreement"
    if "confidence" in reason:
        return "low confidence"
    if "pass threshold" in reason:
        return "score near threshold"
    return "judge error"


def calibrate(
    judge: Judge,
    rubric: Rubric,
    items: list[JudgeItem],
    policy: ReviewPolicy | None = None,
    dataset: str = "",
) -> CalibrationReport:
    """Run the judge over labelled items and summarise how it compares with the labels."""
    policy = policy or ReviewPolicy()
    confusion = ConfusionMatrix()
    decisions: list[Decision] = []
    disagreements: list[Disagreement] = []
    reasons: dict[str, int] = {}
    human: list[str] = []
    machine: list[str] = []
    labelled = 0
    for item in items:
        decision = decide(item, rubric, judge, policy)
        decisions.append(decision)
        for reason in decision.reasons:
            key = _reason_key(reason)
            reasons[key] = reasons.get(key, 0) + 1
        if item.human_label is None:
            continue
        labelled += 1
        if decision.label == "needs_human_review":
            if item.human_label == "pass":
                confusion.review_of_pass += 1
            else:
                confusion.review_of_fail += 1
            continue
        human.append(item.human_label)
        machine.append(decision.label)
        if item.human_label == "pass" and decision.label == "pass":
            confusion.true_pass += 1
        elif item.human_label == "fail" and decision.label == "fail":
            confusion.true_fail += 1
        else:
            if decision.label == "pass":
                confusion.false_pass += 1
            else:
                confusion.false_fail += 1
            disagreements.append(
                Disagreement(
                    item_id=item.id,
                    human=item.human_label,
                    judge=decision.label,
                    score=decision.score,
                    confidence=decision.confidence,
                    note=item.note,
                )
            )
    decided = len(human)
    needs_review = confusion.review_of_pass + confusion.review_of_fail
    return CalibrationReport(
        judge=judge.name,
        rubric=rubric.id,
        dataset=" ".join(dataset.split()),
        policy=policy,
        items=len(items),
        labelled=labelled,
        decided=decided,
        needs_review=needs_review,
        review_rate=needs_review / labelled if labelled else None,
        confusion=confusion,
        agreement=Rate.of(confusion.true_pass + confusion.true_fail, decided),
        kappa=cohens_kappa(human, machine),
        false_positive_rate=Rate.of(
            confusion.false_pass, confusion.false_pass + confusion.true_fail
        ),
        false_negative_rate=Rate.of(
            confusion.false_fail, confusion.false_fail + confusion.true_pass
        ),
        disagreements=disagreements,
        review_reasons=dict(sorted(reasons.items())),
        decisions=decisions,
    )
