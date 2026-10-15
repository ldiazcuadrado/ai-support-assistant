from __future__ import annotations

from collections.abc import Hashable, Sequence

from support_assistant.domain import ScoredChunk
from support_assistant.retrieval.base import Retriever


def reciprocal_rank_fusion[K: Hashable](
    rankings: Sequence[Sequence[K]], k: int = 60
) -> dict[K, float]:
    """Fuse ranked lists; an item scores ``sum(1 / (k + rank))`` across the lists holding it."""
    fused: dict[K, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """Runs several retrievers and merges their results with reciprocal rank fusion.

    The relevance of a fused result is the highest relevance any retriever assigned to it.
    """

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, depth: int = 3) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self._retrievers = list(retrievers)
        self._rrf_k = rrf_k
        self._depth = depth

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        if len(self._retrievers) == 1:
            return self._retrievers[0].search(query, limit)
        results = [r.search(query, limit * self._depth) for r in self._retrievers]
        keyed: dict[tuple[str, int], ScoredChunk] = {}
        relevance: dict[tuple[str, int], float] = {}
        for result in results:
            for item in result:
                key = (item.chunk.source, item.chunk.position)
                keyed.setdefault(key, item)
                relevance[key] = max(relevance.get(key, 0.0), item.relevance)
        fused = reciprocal_rank_fusion(
            [[(i.chunk.source, i.chunk.position) for i in result] for result in results],
            self._rrf_k,
        )
        ranked = sorted(fused, key=lambda key: fused[key], reverse=True)[:limit]
        return [
            ScoredChunk(chunk=keyed[key].chunk, score=fused[key], relevance=relevance[key])
            for key in ranked
        ]
