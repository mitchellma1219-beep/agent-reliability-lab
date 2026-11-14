"""Groundedness: is each statement in the answer supported by what the agent saw?

This is a lexical check, not an entailment model. A sentence counts as supported when enough
of its content words and numbers occur somewhere in the evidence, and at least one single
passage covers a reasonable share of them. A statement whose closest evidence differs in
negation ("are available" against "are not available") is flagged as well. Beyond that the
check does not understand meaning or attribution. See ``docs/evaluator-limitations.md``.
"""

from __future__ import annotations

import re

from pydantic import JsonValue

from ..evidence import collect, known_sources, prose, sentence_passages
from ..instructions import CITATION
from ..models import Case, Evaluation, Finding, Trace
from ..text import content_tokens, overlap, split_sentences
from .base import EvalContext, finding, ratio, verdict

MIN_CLAIM_TOKENS = 3

_META = re.compile(
    r"^(?:i (?:could ?n[o']t|cannot|can't|was unable to|am unable to|do not|don't) |"
    r"sorry|unfortunately|please |could you |can you |which |here is |here are |"
    r"no further action|nothing was changed|completed before the failure|let me know)",
    re.IGNORECASE,
)


_NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|neither|nor|none)\b|n't\b", re.IGNORECASE)


def negated(text: str) -> bool:
    return bool(_NEGATION.search(text))


def claims(output: str, ignore: list[str]) -> list[tuple[str, list[str]]]:
    """Sentences of the output that assert something, each with the sources it cites."""
    found: list[tuple[str, list[str]]] = []
    for sentence in split_sentences(output):
        cited = CITATION.findall(sentence)
        text = CITATION.sub("", sentence).strip()
        if not text or text.endswith("?") or _META.match(text):
            continue
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in ignore):
            continue
        if len(set(content_tokens(text))) < MIN_CLAIM_TOKENS:
            continue
        found.append((text, cited))
    return found


class Groundedness:
    name = "groundedness"

    def applies(self, case: Case, trace: Trace) -> bool:
        check = case.expected.groundedness
        return check is not None and check.enabled

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        check = case.expected.groundedness
        if check is None:
            return verdict(self.name, [])
        passages = collect(case, trace, ctx.context)
        texts = sentence_passages(passages)
        pieces = [(piece.source, set(content_tokens(piece.text))) for piece in texts]
        everything: set[str] = set().union(*(tokens for _, tokens in pieces)) if pieces else set()
        sources = known_sources(trace, ctx.context)

        findings: list[Finding] = []
        report: list[JsonValue] = []
        supported = 0
        found = claims(prose(trace.output), check.ignore_patterns)
        for text, cited in found:
            tokens = set(content_tokens(text))
            coverage = overlap(tokens, everything)
            scores = [overlap(tokens, piece) for _, piece in pieces]
            best = max(scores, default=0.0)
            ok = coverage >= check.token_threshold and best >= check.passage_threshold
            if ok and negated(text) != negated(texts[scores.index(best)].text):
                ok = False
                findings.append(
                    finding(
                        "polarity_mismatch",
                        f"the statement {text[:120]!r} and its closest evidence differ in negation",
                    )
                )
            unknown = sorted(set(cited) - sources)
            if unknown:
                ok = False
                findings.append(
                    finding("unknown_citation", f"cites a source the agent never saw: {unknown}")
                )
            elif cited and ok:
                cited_best = max(
                    (overlap(tokens, piece) for source, piece in pieces if source in cited),
                    default=0.0,
                )
                if cited_best < check.passage_threshold:
                    ok = False
                    findings.append(
                        finding(
                            "citation_mismatch",
                            f"the cited source {cited} does not support: {text[:120]!r}",
                        )
                    )
            if not ok and not unknown and coverage < check.token_threshold:
                missing = sorted(tokens - everything)[:8]
                findings.append(
                    finding(
                        "unsupported_claim",
                        f"not supported by the evidence: {text[:120]!r} (no source for: "
                        f"{', '.join(missing)})",
                    )
                )
            elif not ok and not unknown and best < check.passage_threshold:
                findings.append(
                    finding(
                        "unsupported_claim",
                        f"no single passage supports: {text[:120]!r} (best passage covers "
                        f"{best:.0%} of it)",
                    )
                )
            supported += ok
            report.append(
                {
                    "claim": text[:200],
                    "coverage": round(coverage, 3),
                    "best_passage": round(best, 3),
                }
            )
        score = ratio(supported, len(found))
        if score >= check.min_supported:
            findings = [
                item.model_copy(update={"severity": "warning"})
                if item.code == "unsupported_claim"
                else item
                for item in findings
            ]
        return verdict(
            self.name,
            findings,
            score=score,
            details={"claims": len(found), "supported": supported, "by_claim": report[:20]},
        )
