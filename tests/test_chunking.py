from __future__ import annotations

from itertools import pairwise

import pytest

from support_assistant.domain import LoadedDocument, Page
from support_assistant.ingestion import chunk_document, chunk_text


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("  \n\n  ", size=100, overlap=10) == []


def test_chunks_respect_size_limit() -> None:
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(20))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = "\n\n".join(f"Sentence number {i} is here and says something." for i in range(30))
    chunks = chunk_text(text, size=200, overlap=60)
    for previous, current in pairwise(chunks):
        head = current.split("\n\n")[0]
        assert head and head in previous


def test_zero_overlap_does_not_repeat_text() -> None:
    chunks = chunk_text("\n\n".join(f"unit{i} " + "x" * 40 for i in range(10)), size=120, overlap=0)
    joined = "\n\n".join(chunks)
    assert all(joined.count(f"unit{i} ") == 1 for i in range(10))


def test_long_paragraph_is_split_on_sentences() -> None:
    text = " ".join(f"This is sentence {i}." for i in range(100))
    chunks = chunk_text(text, size=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_unbroken_token_is_hard_split() -> None:
    chunks = chunk_text("a" * 1000, size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", size=size, overlap=overlap)


def test_chunk_document_carries_metadata_and_positions() -> None:
    doc = LoadedDocument(
        source="guide.pdf",
        title="Guide",
        pages=(Page("First page text.", 1), Page("Second page text.", 2)),
    )
    chunks = chunk_document(doc, size=100, overlap=10)
    assert [(c.page, c.position) for c in chunks] == [(1, 0), (2, 1)]
    assert {c.source for c in chunks} == {"guide.pdf"}
    assert {c.title for c in chunks} == {"Guide"}


def test_word_longer_than_size_inside_text_is_split() -> None:
    chunks = chunk_text("short " + "z" * 250 + " tail words here", size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).count("z") >= 250
