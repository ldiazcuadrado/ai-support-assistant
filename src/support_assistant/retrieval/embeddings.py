from __future__ import annotations

import math
from array import array
from collections.abc import Sequence

from support_assistant.domain import ScoredChunk
from support_assistant.errors import ConfigError
from support_assistant.retrieval.base import Embedder
from support_assistant.storage import KnowledgeStore


class FastEmbedEmbedder:  # pragma: no cover - requires the optional dependency and a model download
    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ConfigError(
                "EMBEDDINGS_BACKEND=fastembed requires the extra: "
                "pip install 'ai-support-assistant[embeddings]'"
            ) from exc
        self._name = model_name
        self._model = TextEmbedding(model_name=model_name)

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._model.embed(list(texts))]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class VectorRetriever:
    """Exact nearest-neighbour search over vectors stored in SQLite.

    Vectors are scanned linearly, which is fine for documentation-sized corpora (thousands of
    chunks). Swap in another ``Retriever`` for larger ones.
    """

    def __init__(self, store: KnowledgeStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        stored = self._store.load_embeddings(self._embedder.name)
        if not stored or not query.strip():
            return []
        (query_vec,) = self._embedder.embed([query])
        scored: list[tuple[int, float]] = [
            (chunk_id, cosine(query_vec, _as_floats(vec))) for chunk_id, vec in stored
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[:limit]
        chunks = self._store.get_many([chunk_id for chunk_id, _ in top])
        return [
            ScoredChunk(chunk=chunks[chunk_id], score=sim, relevance=max(0.0, min(1.0, sim)))
            for chunk_id, sim in top
            if chunk_id in chunks
        ]


def _as_floats(vec: array[float]) -> Sequence[float]:
    return vec
