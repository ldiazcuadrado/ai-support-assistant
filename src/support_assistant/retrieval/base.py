from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from support_assistant.domain import ScoredChunk


class Retriever(Protocol):
    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        """Return up to ``limit`` chunks, best first."""
        ...


class Embedder(Protocol):
    """Turns text into dense vectors. ``name`` identifies the model in the vector store."""

    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
