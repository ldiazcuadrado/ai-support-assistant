from __future__ import annotations

import re
import zlib
from collections.abc import Sequence

import pytest

from support_assistant.domain import Chunk, ScoredChunk
from support_assistant.factory import Container
from support_assistant.retrieval import (
    HybridRetriever,
    LexicalRetriever,
    VectorRetriever,
    reciprocal_rank_fusion,
)
from support_assistant.retrieval.embeddings import cosine
from support_assistant.retrieval.text import coverage, stem, stems, tokens


class HashingEmbedder:
    """Deterministic bag-of-words embedder: enough to test the vector path offline."""

    name = "hashing-test"
    dims = 64

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dims
            for word in re.findall(r"\w+", text.lower()):
                vec[zlib.crc32(stem(word).encode()) % self.dims] += 1.0
            vectors.append(vec)
        return vectors


def _top(c: Container, query: str) -> str:
    hits = LexicalRetriever(c.knowledge).search(query, 3)
    assert hits, query
    return hits[0].chunk.source


@pytest.mark.parametrize(
    ("query", "source"),
    [
        ("What does a puncture repair cost?", "services-and-pricing.md"),
        ("what time do you open on Saturday", "opening-hours.md"),
        ("Do you repair Bosch motors on electric bikes?", "ebike-service.html"),
        ("how long is the workmanship guarantee", "warranty-and-returns.md"),
        ("Do you deliver bikes to my house?", "faq.txt"),
    ],
)
def test_lexical_ranking_finds_the_right_document(
    container: Container, query: str, source: str
) -> None:
    assert _top(container, query) == source


def test_relevance_separates_on_topic_from_off_topic(container: Container) -> None:
    retriever = LexicalRetriever(container.knowledge)
    on_topic = retriever.search("How much does a full service cost?", 3)[0]
    off_topic = retriever.search("do you sell pizza and sushi", 3)
    assert on_topic.relevance >= 0.5
    assert all(h.relevance < 0.4 for h in off_topic)


def test_query_of_only_stopwords_returns_nothing(container: Container) -> None:
    assert LexicalRetriever(container.knowledge).search("what is the", 3) == []


def test_query_with_fts_syntax_is_treated_as_text(container: Container) -> None:
    assert LexicalRetriever(container.knowledge).search('NEAR( " OR * AND', 3) is not None


def test_stemming_matches_inflections() -> None:
    assert stem("tyres") == stem("tyre")
    assert stem("repairing") == stem("repairs") == stem("repair")
    assert tokens("The price of the price") == ["price"]
    assert coverage(stems("tyre price"), "Tyres cost money, see price list") == 1.0
    assert coverage(set(), "anything") == 0.0


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == fused["b"]
    assert fused["c"] > fused["d"] or fused["c"] == fused["d"]
    only_first = reciprocal_rank_fusion([["x", "y"], ["z"]])
    assert only_first["x"] > only_first["y"]


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_vector_retriever_and_hybrid_fusion(make_container: type) -> None:
    from support_assistant.ingestion import IngestPipeline
    from tests.conftest import KNOWLEDGE_BASE

    c: Container = make_container()
    embedder = HashingEmbedder()
    IngestPipeline(c.knowledge, 400, 60, embedder).ingest_path(KNOWLEDGE_BASE)

    vector = VectorRetriever(c.knowledge, embedder)
    hits = vector.search("storage fee for uncollected bikes", 3)
    assert hits[0].chunk.source == "booking-and-turnaround.md"
    assert 0.0 <= hits[0].relevance <= 1.0

    hybrid = HybridRetriever([LexicalRetriever(c.knowledge), vector])
    fused = hybrid.search("storage fee for uncollected bikes", 3)
    assert fused[0].chunk.source == "booking-and-turnaround.md"
    assert len(fused) <= 3
    lexical = LexicalRetriever(c.knowledge).search("storage fee for uncollected bikes", 3)
    assert fused[0].relevance >= lexical[0].relevance


def test_vector_retriever_without_vectors_is_empty(container: Container) -> None:
    assert VectorRetriever(container.knowledge, HashingEmbedder()).search("anything", 3) == []


class _Static:
    def __init__(self, items: list[ScoredChunk]) -> None:
        self._items = items

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        return self._items[:limit]


def _sc(position: int, relevance: float) -> ScoredChunk:
    return ScoredChunk(Chunk("s", "t", None, position, f"text {position}"), 1.0, relevance)


def test_hybrid_ranks_by_fusion_and_keeps_best_relevance() -> None:
    first = _Static([_sc(0, 0.3), _sc(1, 0.9), _sc(2, 0.1)])
    second = _Static([_sc(1, 0.2)])
    result = HybridRetriever([first, second]).search("q", 2)
    assert [r.chunk.position for r in result] == [1, 0]
    assert [r.relevance for r in result] == [0.9, 0.3]


def test_hybrid_requires_a_retriever_and_passes_through_a_single_one() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])
    only = _Static([_sc(0, 0.5)])
    assert HybridRetriever([only]).search("q", 5) == only.search("q", 5)
