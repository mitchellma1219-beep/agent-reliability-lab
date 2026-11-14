"""What the agent actually saw: the basis for groundedness and hallucination checks.

Evidence is the request, the instructions, the documents handed over up front, and everything
tools returned (results of successful calls and the messages of failed ones). Arguments the
agent *sent* to tools are not evidence: the agent wrote those itself.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .models import Case, Document, Trace
from .text import split_sentences


@dataclass(frozen=True)
class Passage:
    source: str
    text: str


def flatten(value: Any) -> str:
    """Readable text for a JSON value: ``{"order_id": "A-1"}`` -> ``order id: A-1``."""
    if isinstance(value, dict):
        pairs = [(str(key).replace("_", " "), flatten(item)) for key, item in value.items()]
        return ". ".join(f"{key}: {text}" for key, text in pairs if text)
    if isinstance(value, list):
        return ". ".join(text for text in (flatten(item) for item in value) if text)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _search_hits(result: Any) -> list[tuple[str, str]] | None:
    """(id, text) for a search-style result, or ``None`` if the result has another shape."""
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        return None
    hits: list[tuple[str, str]] = []
    for item in result["results"]:
        if not (isinstance(item, dict) and isinstance(item.get("text"), str)):
            return None
        title = item.get("title") if isinstance(item.get("title"), str) else ""
        hits.append(
            (str(item.get("id", "")), f"{title}. {item['text']}" if title else item["text"])
        )
    return hits


def collect(case: Case, trace: Trace, context: Sequence[Document] = ()) -> list[Passage]:
    """All evidence passages for one run, in the order the agent met them."""
    passages = [Passage("request", case.input)]
    passages += [Passage(f"instruction:{item.id}", item.text) for item in case.instructions]
    passages += [
        Passage(doc.id, f"{doc.title}. {doc.text}" if doc.title else doc.text) for doc in context
    ]
    for call in trace.tool_calls:
        label = f"{call.tool}#{call.index}"
        if not call.ok:
            if call.error is not None:
                passages.append(Passage(label, f"{call.error.code}: {call.error.message}"))
            continue
        hits = _search_hits(call.result)
        if hits is not None:
            passages += [Passage(doc_id or label, text) for doc_id, text in hits]
            continue
        words = call.tool.replace("_", " ")
        text = flatten(call.result)
        if isinstance(call.result, str):
            try:
                text = flatten(json.loads(call.result))
            except (ValueError, RecursionError):
                text = call.result
        passages.append(Passage(label, f"{words}. {text}"))
    return passages


def known_sources(trace: Trace, context: Sequence[Document] = ()) -> frozenset[str]:
    """Ids of documents the agent was given or retrieved (valid targets for a citation)."""
    ids = {doc.id for doc in context}
    for call in trace.tool_calls:
        hits = _search_hits(call.result) if call.ok else None
        ids.update(doc_id for doc_id, _ in hits or [] if doc_id)
    return frozenset(ids)


def sentence_passages(passages: Sequence[Passage]) -> list[Passage]:
    """Split document-like passages into sentences; keep each tool result whole as well."""
    pieces: list[Passage] = []
    for passage in passages:
        sentences = split_sentences(passage.text)
        pieces.extend(Passage(passage.source, sentence) for sentence in sentences)
        if len(sentences) > 1 and "#" in passage.source:
            pieces.append(passage)
    return pieces


def sink_text(trace: Trace) -> str:
    """Everything the agent passed to sink tools, as one JSON-ish string."""
    return "\n".join(
        json.dumps(call.arguments, ensure_ascii=False, default=str)
        for call in trace.tool_calls
        if call.sink
    )


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _string_values(item)]
    if isinstance(value, list):
        return [text for item in value for text in _string_values(item)]
    return []


def prose(output: str) -> str:
    """The readable text of an answer. For a JSON answer, its string values one per line."""
    stripped = output.strip()
    if stripped[:1] not in "{[":
        return output
    try:
        payload = json.loads(stripped)
    except (ValueError, RecursionError):
        return output
    return "\n".join(_string_values(payload))
