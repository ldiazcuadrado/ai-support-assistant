"""Turn documents into chunks and store them."""

from support_assistant.ingestion.chunking import chunk_document, chunk_text
from support_assistant.ingestion.loaders import FetchedPage, Fetcher, http_fetcher
from support_assistant.ingestion.pipeline import IngestPipeline, IngestReport

__all__ = [
    "FetchedPage",
    "Fetcher",
    "IngestPipeline",
    "IngestReport",
    "chunk_document",
    "chunk_text",
    "http_fetcher",
]
