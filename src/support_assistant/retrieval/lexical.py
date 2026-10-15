from __future__ import annotations

from support_assistant.domain import ScoredChunk
from support_assistant.retrieval import text
from support_assistant.storage import KnowledgeStore


class LexicalRetriever:
    """BM25 ranking through SQLite FTS5.

    FTS5 scores are unbounded and depend on corpus statistics, so confidence is derived from the
    share of query terms the chunk contains instead.
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        terms = text.tokens(query)
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        query_stems = {text.stem(term) for term in terms}
        return [
            ScoredChunk(
                chunk=chunk,
                score=score,
                relevance=text.coverage(query_stems, f"{chunk.title} {chunk.text}"),
            )
            for chunk, score in self._store.search_fts(match, limit)
        ]
