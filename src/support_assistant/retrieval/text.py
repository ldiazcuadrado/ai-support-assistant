from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "after",
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
        "been",
        "before",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "get",
        "got",
        "had",
        "has",
        "have",
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
        "my",
        "of",
        "on",
        "or",
        "our",
        "out",
        "please",
        "s",
        "should",
        "so",
        "some",
        "t",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "tell",
        "know",
        "want",
        "need",
        "much",
        "many",
    ]
)


def stem(token: str) -> str:
    """Crude suffix stripper, adequate for matching "tyre", "tyres" and "repairing" to roots."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def tokens(text: str) -> list[str]:
    """Lower-cased content words, in order of appearance and without duplicates."""
    seen: dict[str, None] = {}
    for token in _TOKEN.findall(text.lower()):
        if len(token) > 1 and token not in STOPWORDS:
            seen.setdefault(token)
    return list(seen)


def stems(text: str) -> set[str]:
    return {stem(token) for token in tokens(text)}


def coverage(query_stems: set[str], text: str) -> float:
    """Fraction of query terms that appear in ``text`` (after stemming)."""
    if not query_stems:
        return 0.0
    return len(query_stems & stems(text)) / len(query_stems)
