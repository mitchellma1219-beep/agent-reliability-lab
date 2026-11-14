"""Hallucination detection: specifics in the answer that have no source.

Four kinds of "atoms" are extracted from the output and looked up in the evidence:

* **numbers** (a number counts as supported if it is a rounding of a number in the evidence;
  bare integers up to ten are skipped as counts unless a unit such as "days" or "%" follows);
* **identifiers** such as ``A-1001`` or ``SKU-100`` (tokens mixing letters and digits);
* **addresses**: e-mail addresses and URLs;
* **claimed actions**: sentences such as "the refund has been issued" when no successful
  call to the corresponding tool exists in the trace.

It does not judge qualitative statements; that is what ``groundedness`` attempts.
"""

from __future__ import annotations

import re

from ..evidence import collect
from ..instructions import CITATION
from ..models import Case, Evaluation, Finding, Trace
from ..text import split_sentences
from .base import EvalContext, finding, verdict

_NUMBER = re.compile(r"(?<![\w.,-])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)")
_IDENTIFIER = re.compile(
    r"(?<![\w@./-])(?=[\w-]*[A-Za-z])(?=[\w-]*\d)[A-Za-z0-9][\w-]{2,}(?![\w@])"
)
_UNIT_SUFFIX = re.compile(r"^\d+(?:st|nd|rd|th|[a-z]{1,2})$", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s<>\")\]]+", re.IGNORECASE)
_UNIT = re.compile(
    r"\s?(?:%|percent\b|(?:business |calendar |working )?(?:day|week|month|year|hour|minute|"
    r"second)s?\b|(?:usd|eur|gbp|ms)\b|dollars?\b|euros?\b)",
    re.IGNORECASE,
)
_LIST_MARKER = re.compile(r"^\s*\d{1,3}[.)]\s+", re.MULTILINE)
_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|unable|failed|without|pending|will|would|can|could|should|"
    r"may|might|if|once|before)\b|n't\b",
    re.IGNORECASE,
)


def _numbers(text: str) -> list[tuple[float, int, str, bool]]:
    """(value, decimals, original text, has a unit) for each number in ``text``."""
    found: list[tuple[float, int, str, bool]] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        unit = bool(_UNIT.match(text[match.end() : match.end() + 24]))
        found.append((value, decimals, match.group(0), unit))
    return found


def _strip_addresses(text: str) -> str:
    return _URL.sub(" ", _EMAIL.sub(" ", text))


class Hallucination:
    name = "hallucination"

    def applies(self, case: Case, trace: Trace) -> bool:
        check = case.expected.hallucination
        return check is not None and check.enabled

    def evaluate(self, case: Case, trace: Trace, ctx: EvalContext) -> Evaluation:
        check = case.expected.hallucination
        if check is None:
            return verdict(self.name, [])
        evidence = "\n".join(passage.text for passage in collect(case, trace, ctx.context))
        folded = evidence.casefold()
        allowed = {item.casefold() for item in check.allow}
        output = _LIST_MARKER.sub("", CITATION.sub(" ", trace.output))
        findings: list[Finding] = []
        atoms = 0

        addresses = [*_EMAIL.findall(output), *_URL.findall(output)]
        for address in dict.fromkeys(addresses):
            atoms += 1
            cleaned = address.rstrip(".,;:!?")
            if cleaned.casefold() not in folded and cleaned.casefold() not in allowed:
                findings.append(
                    finding("unsupported_address", f"address has no source: {cleaned[:100]}")
                )

        body = _strip_addresses(output)
        if check.check_identifiers:
            known = {match.casefold() for match in _IDENTIFIER.findall(_strip_addresses(evidence))}
            for identifier in dict.fromkeys(_IDENTIFIER.findall(body)):
                if _UNIT_SUFFIX.match(identifier):
                    continue
                atoms += 1
                key = identifier.casefold()
                if key not in known and key not in folded and key not in allowed:
                    findings.append(
                        finding("unsupported_identifier", f"identifier has no source: {identifier}")
                    )

        if check.check_numbers:
            source_numbers = [item[0] for item in _numbers(_strip_addresses(evidence))]
            for value, decimals, raw, unit in dict.fromkeys(_numbers(_IDENTIFIER.sub(" ", body))):
                if decimals == 0 and 0 <= value <= check.ignore_integers_up_to and not unit:
                    continue
                atoms += 1
                if raw.casefold() in allowed:
                    continue
                tolerance = 0.5 * 10**-decimals + 1e-9
                if not any(abs(value - source) <= tolerance for source in source_numbers):
                    findings.append(finding("unsupported_number", f"number has no source: {raw}"))

        if check.check_actions:
            for sentence in split_sentences(trace.output):
                if _NEGATION.search(sentence):
                    continue
                for tool, spec in ctx.tools.items():
                    if not spec.claim_patterns or trace.successful_calls(tool):
                        continue
                    if any(re.search(p, sentence, re.IGNORECASE) for p in spec.claim_patterns):
                        atoms += 1
                        findings.append(
                            finding(
                                "claimed_action_not_performed",
                                f"the answer says {sentence[:120]!r} but no successful "
                                f"call to {tool} was made",
                            )
                        )

        return verdict(
            self.name,
            findings,
            score=1.0 if not atoms else max(0.0, 1.0 - len(findings) / max(atoms, len(findings))),
            details={"atoms_checked": atoms, "unsupported": len(findings)},
        )
