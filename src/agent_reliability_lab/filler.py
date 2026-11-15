"""Deterministic filler documents for long-context cases.

The text is bland, digit-free operational prose assembled from a fixed vocabulary, so it adds
length and lexical noise without adding facts that could accidentally answer a question.
"""

from __future__ import annotations

from .faults import uniform
from .models import Case, Document, FillerSpec

_VOCABULARY = [
    "quarterly",
    "logistics",
    "warehouse",
    "review",
    "meeting",
    "agenda",
    "supplier",
    "onboarding",
    "checklist",
    "facility",
    "maintenance",
    "schedule",
    "regional",
    "planning",
    "committee",
    "notes",
    "summary",
    "inventory",
    "audit",
    "procedure",
    "training",
    "session",
    "calendar",
    "holiday",
    "roster",
    "office",
    "seating",
    "parking",
    "newsletter",
    "archive",
    "volunteer",
    "programme",
    "cafeteria",
    "menu",
    "rotation",
    "printer",
    "supplies",
    "badge",
    "renewal",
    "visitor",
    "register",
    "cleaning",
    "contract",
    "recycling",
    "initiative",
    "team",
    "outing",
    "mentoring",
    "circle",
    "onboarding",
    "buddy",
    "handbook",
    "glossary",
    "travel",
    "guidance",
    "expense",
    "categories",
    "branding",
    "palette",
    "typography",
    "template",
    "slide",
    "library",
    "photo",
    "guidelines",
    "press",
    "contacts",
    "building",
    "access",
    "shuttle",
    "timetable",
    "wellness",
    "workshop",
    "ergonomics",
    "survey",
    "feedback",
    "forum",
]


def filler_documents(spec: FillerSpec) -> list[Document]:
    documents: list[Document] = []
    for number in range(spec.count):
        words = [
            _VOCABULARY[int(uniform(spec.seed, number, position) * len(_VOCABULARY))]
            for position in range(spec.words)
        ]
        sentences = [
            " ".join(words[start : start + 12]).capitalize() + "."
            for start in range(0, len(words), 12)
        ]
        documents.append(
            Document(
                id=f"filler-{number + 1:03d}",
                title=f"Internal memo {number + 1}",
                text=" ".join(sentences),
            )
        )
    return documents


def build_context(case: Case) -> list[Document]:
    """The documents handed to the agent up front: authored ones with filler in between.

    Filler is split evenly into the gaps *before* each authored document, so the authored
    documents keep their order and the last authored document stays last.
    """
    if case.context_filler is None:
        return list(case.context)
    filler = filler_documents(case.context_filler)
    if not case.context:
        return filler
    gaps = len(case.context)
    share, extra = divmod(len(filler), gaps)
    merged: list[Document] = []
    cursor = 0
    for position, document in enumerate(case.context):
        take = share + (1 if position < extra else 0)
        merged.extend(filler[cursor : cursor + take])
        cursor += take
        merged.append(document)
    return merged
