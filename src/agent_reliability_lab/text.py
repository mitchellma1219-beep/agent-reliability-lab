"""Small, deterministic text utilities shared by tools, simulated agents and evaluators.

Nothing here understands language. Tokens are lower-cased words and normalised numbers, with
a crude suffix stripper so that ``refund``, ``refunds`` and ``refunded`` compare equal.
"""

from __future__ import annotations

import math
import re
import unicodedata

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "please",
        "tell",
        "know",
        "want",
        "need",
        "according",
        "much",
        "many",
        "long",
        "per",
        "via",
    ]
)

_TOKEN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-_]")
_BIDI = re.compile(r"[\u202a-\u202e\u2066-\u2069\u200e\u200f]")


def normalise_number(text: str) -> str | None:
    """Canonical text of a number such as ``1,250.50`` -> ``1250.5``; ``None`` if not a number."""
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return str(int(value)) if value == int(value) and abs(value) < 1e15 else repr(value)


def stem(word: str) -> str:
    """Strip common English suffixes. Consistency matters here, not linguistic correctness."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif (len(word) > 4 and word.endswith("ed")) or (len(word) > 4 and word.endswith("es")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Stemmed content words and normalised numbers, in order, without stop words."""
    found: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw[0].isdigit():
            number = normalise_number(raw)
            if number is not None:
                found.append(number)
            continue
        if raw in STOPWORDS or len(raw) < 2:
            continue
        found.append(stem(raw))
    return found


def split_sentences(text: str) -> list[str]:
    """Split on line breaks and sentence punctuation; list markers are removed."""
    sentences: list[str] = []
    for line in text.splitlines():
        stripped = _BULLET.sub("", line).strip()
        for piece in _SENTENCE_END.split(stripped):
            cleaned = piece.strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


def word_count(text: str) -> int:
    return len(text.split())


def overlap(query_tokens: set[str], passage_tokens: set[str]) -> float:
    """Share of the query tokens found in the passage (0 when the query is empty)."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & passage_tokens) / len(query_tokens)


def strip_control(text: str) -> str:
    """Remove ANSI escapes, control characters and bidi overrides from untrusted text.

    Used before printing agent output to a terminal or writing it into a report, so that an
    agent cannot repaint the screen or visually reorder what a reviewer reads.
    """
    text = _ANSI.sub("", text)
    text = _BIDI.sub("", text)
    return _CONTROL.sub("", text)


def squash(text: str) -> str:
    """Lower-case and keep only letters and digits. Used for obfuscation-tolerant matching."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in folded if char.isalnum())


def truncate(text: str, limit: int, marker: str = " …[truncated]") -> str:
    return text if len(text) <= limit else text[:limit] + marker
