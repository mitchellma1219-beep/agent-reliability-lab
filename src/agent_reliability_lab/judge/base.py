"""Judge protocol and the rule that routes uncertain verdicts to a person.

A judge scores one answer against a rubric. Judges are optional: nothing in the default
evaluation path uses one, and the only backend that calls a model is off unless explicitly
enabled. A judge's verdict is treated as an *opinion with an error rate*: measure that rate
with :mod:`.calibration` before relying on it, and let :func:`decide` hold back the verdicts
it should not be trusted on.
"""

from __future__ import annotations

from typing import Protocol

from ..errors import JudgeError
from .models import Decision, FinalLabel, JudgeItem, JudgeVerdict, ReviewPolicy, Rubric


class Judge(Protocol):
    name: str

    def judge(self, item: JudgeItem, rubric: Rubric, sample: int = 0) -> JudgeVerdict:
        """Score one item. ``sample`` numbers repeated readings of the same item."""
        ...


def decide(
    item: JudgeItem, rubric: Rubric, judge: Judge, policy: ReviewPolicy | None = None
) -> Decision:
    """Judge an item and apply the review policy.

    The result is ``needs_human_review`` when any of the following holds:

    * the judge raised an error or returned something unusable;
    * repeated readings of the same item disagree on the label;
    * the lowest confidence is below ``policy.min_confidence``;
    * the mean score is within ``policy.ambiguity_margin`` of the rubric's pass threshold.
    """
    policy = policy or ReviewPolicy()
    verdicts: list[JudgeVerdict] = []
    reasons: list[str] = []
    for sample in range(policy.samples):
        try:
            verdicts.append(judge.judge(item, rubric, sample))
        except JudgeError as exc:
            reasons.append(f"the judge could not produce a verdict: {exc}")
            break
    if not verdicts:
        return Decision(item_id=item.id, label="needs_human_review", reasons=reasons)

    score = sum(verdict.score for verdict in verdicts) / len(verdicts)
    confidence = min(verdict.confidence for verdict in verdicts)
    passes = sum(1 for verdict in verdicts if verdict.label == "pass")
    if 0 < passes < len(verdicts):
        reasons.append(
            f"the judge disagreed with itself ({passes} of {len(verdicts)} readings said pass)"
        )
    if confidence < policy.min_confidence:
        reasons.append(f"confidence {confidence:.2f} is below {policy.min_confidence:.2f}")
    if abs(score - rubric.pass_threshold) < policy.ambiguity_margin:
        reasons.append(
            f"score {score:.2f} is within {policy.ambiguity_margin:.2f} of the pass threshold "
            f"{rubric.pass_threshold:.2f}"
        )
    label: FinalLabel = "pass" if passes * 2 > len(verdicts) else "fail"
    if reasons:
        label = "needs_human_review"
    return Decision(
        item_id=item.id,
        label=label,
        score=score,
        confidence=confidence,
        reasons=reasons,
        verdicts=verdicts,
    )
