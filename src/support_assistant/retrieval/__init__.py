"""Retrieval backends. Every backend implements the ``Retriever`` protocol."""

from support_assistant.retrieval.base import Embedder, Retriever
from support_assistant.retrieval.embeddings import FastEmbedEmbedder, VectorRetriever
from support_assistant.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from support_assistant.retrieval.lexical import LexicalRetriever

__all__ = [
    "Embedder",
    "FastEmbedEmbedder",
    "HybridRetriever",
    "LexicalRetriever",
    "Retriever",
    "VectorRetriever",
    "reciprocal_rank_fusion",
]
