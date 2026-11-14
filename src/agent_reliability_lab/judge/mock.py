"""Deterministic judges for tests and demonstrations. They do not call any model.

:class:`MockJudge` scores an answer by word overlap with the reference answer. It is a
stand-in that makes the judging pipeline testable, and it is deliberately imperfect: it
passes an answer that reuses the reference's words around a wrong number and fails a correct
paraphrase. Calibration is expected to expose exactly that.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..errors import JudgeError
from ..faults import uniform
from ..text import content_tokens
from .models import JudgeItem, JudgeVerdict, Label, Rubric


def token_f1(candidate: str, reference: str) -> float:
    """Harmonic mean of precision and recall over distinct content tokens."""
    left, right = set(content_tokens(candidate)), set(content_tokens(reference))
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if shared == 0:
        return 0.0
    precision, recall = shared / len(left), shared / len(right)
    return 2 * precision * recall / (precision + recall)


class MockJudge:
    """Scores by token overlap with the reference; confidence grows with the margin.

    ``noise`` adds a reproducible per-reading jitter to the score, which lets tests exercise
    the "judge disagrees with itself" rule.
    """

    def __init__(self, name: str = "mock-judge", noise: float = 0.0, seed: int = 0) -> None:
        if not 0.0 <= noise <= 1.0:
            raise JudgeError("noise must be between 0 and 1")
        self.name = name
        self._noise = noise
        self._seed = seed

    def judge(self, item: JudgeItem, rubric: Rubric, sample: int = 0) -> JudgeVerdict:
        if item.reference is None:
            raise JudgeError(f"item {item.id!r} has no reference answer for the mock judge")
        score = token_f1(item.output, item.reference)
        if self._noise:
            jitter = (uniform(self._seed, item.id, sample) - 0.5) * 2.0 * self._noise
            score = min(1.0, max(0.0, score + jitter))
        margin = abs(score - rubric.pass_threshold)
        label: Label = "pass" if score >= rubric.pass_threshold else "fail"
        return JudgeVerdict(
            item_id=item.id,
            label=label,
            score=score,
            confidence=min(1.0, 0.5 + 2.0 * margin),
            rationale=f"token overlap with the reference answer: {score:.2f}",
            criteria={criterion.id: score for criterion in rubric.criteria},
            judge=self.name,
        )


class ScriptedJudge:
    """Returns verdicts from a table: ``{item_id: (score, confidence)}``.

    A value may also be a list of pairs, one per repeated reading. Items missing from the
    table raise :class:`JudgeError`, which the review policy turns into a human review.
    """

    def __init__(
        self,
        table: Mapping[str, tuple[float, float] | list[tuple[float, float]]],
        name: str = "scripted-judge",
    ) -> None:
        self.name = name
        self._table = dict(table)

    def judge(self, item: JudgeItem, rubric: Rubric, sample: int = 0) -> JudgeVerdict:
        entry = self._table.get(item.id)
        if entry is None:
            raise JudgeError(f"no scripted verdict for item {item.id!r}")
        score, confidence = entry[sample % len(entry)] if isinstance(entry, list) else entry
        label: Label = "pass" if score >= rubric.pass_threshold else "fail"
        return JudgeVerdict(
            item_id=item.id, label=label, score=score, confidence=confidence, judge=self.name
        )
