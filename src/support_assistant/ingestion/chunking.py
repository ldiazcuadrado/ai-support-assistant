from __future__ import annotations

import re

from support_assistant.domain import Chunk, LoadedDocument

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_INLINE_SPACE = re.compile(r"[ \t]+")
_WHITESPACE = re.compile(r"\s")


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split ``text`` into chunks of at most ``size`` characters.

    Paragraph boundaries are preferred, then sentences, then words. Each chunk after the first
    starts with up to ``overlap`` characters carried over from the end of the previous one,
    trimmed to a word boundary.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be positive and overlap must be in [0, size)")

    chunks: list[str] = []
    buffer = ""
    for unit in _units(text, size):
        candidate = f"{buffer}\n\n{unit}" if buffer else unit
        if len(candidate) <= size:
            buffer = candidate
            continue
        chunks.append(buffer)
        tail = _tail(buffer, overlap)
        carried = f"{tail}\n\n{unit}" if tail else unit
        buffer = carried if len(carried) <= size else unit
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_document(doc: LoadedDocument, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        for text in chunk_text(page.text, size, overlap):
            chunks.append(
                Chunk(
                    source=doc.source,
                    title=doc.title,
                    page=page.number,
                    position=len(chunks),
                    text=text,
                )
            )
    return chunks


def _units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for raw in _PARAGRAPH_BREAK.split(text):
        paragraph = "\n".join(
            line for line in (_INLINE_SPACE.sub(" ", ln).strip() for ln in raw.splitlines()) if line
        )
        if paragraph:
            units.extend(_fit(paragraph, size))
    return units


def _fit(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BREAK.split(paragraph):
        for part in _split_words(sentence, size):
            candidate = f"{current} {part}" if current else part
            if len(candidate) <= size:
                current = candidate
            else:
                pieces.append(current)
                current = part
    if current:
        pieces.append(current)
    return pieces


def _split_words(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    current = ""
    for token in text.split():
        word = token
        while len(word) > size:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:size])
            word = word[size:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= size:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)
    return parts


def _tail(text: str, overlap: int) -> str:
    if overlap == 0:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    if not text[-overlap - 1].isspace():
        match = _WHITESPACE.search(tail)
        tail = tail[match.end() :] if match else ""
    return tail.strip()
