from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from support_assistant.domain import LoadedDocument
from support_assistant.errors import IngestError
from support_assistant.ingestion.chunking import chunk_document
from support_assistant.ingestion.loaders import (
    Fetcher,
    discover_files,
    load_file,
    load_page,
    sitemap_urls,
)
from support_assistant.retrieval.base import Embedder
from support_assistant.storage import KnowledgeStore

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64


@dataclass(slots=True)
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


class IngestPipeline:
    def __init__(
        self,
        store: KnowledgeStore,
        chunk_size: int,
        chunk_overlap: int,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._size = chunk_size
        self._overlap = chunk_overlap
        self._embedder = embedder

    def ingest_path(self, path: Path) -> IngestReport:
        report = IngestReport()
        for file in discover_files(path):
            try:
                self._add(load_file(file, path), report)
            except IngestError as exc:
                self._skip(report, file.as_posix(), exc)
        return report

    def ingest_urls(self, urls: Iterable[str], fetch: Fetcher) -> IngestReport:
        report = IngestReport()
        for url in urls:
            try:
                self._add(load_page(fetch(url)), report)
            except IngestError as exc:
                self._skip(report, url, exc)
        return report

    def ingest_sitemap(self, url: str, fetch: Fetcher, limit: int = 200) -> IngestReport:
        return self.ingest_urls(sitemap_urls(url, fetch, limit), fetch)

    def _add(self, doc: LoadedDocument, report: IngestReport) -> None:
        chunks = chunk_document(doc, self._size, self._overlap)
        ids = self._store.replace_source(doc.source, chunks)
        if self._embedder is not None:
            self._embed(ids, [c.text for c in chunks])
        report.documents += 1
        report.chunks += len(chunks)
        logger.info("ingested", extra={"source": doc.source, "chunks": len(chunks)})

    def _embed(self, ids: list[int], texts: list[str]) -> None:
        assert self._embedder is not None
        for start in range(0, len(ids), _EMBED_BATCH):
            batch_ids = ids[start : start + _EMBED_BATCH]
            vectors = self._embedder.embed(texts[start : start + _EMBED_BATCH])
            self._store.store_embeddings(
                self._embedder.name, dict(zip(batch_ids, vectors, strict=True))
            )

    @staticmethod
    def _skip(report: IngestReport, source: str, exc: IngestError) -> None:
        report.skipped.append((source, str(exc)))
        logger.warning("skipped", extra={"source": source, "reason": str(exc)})
