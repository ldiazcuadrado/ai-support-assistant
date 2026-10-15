"""Core value types shared by every layer. This package performs no I/O."""

from support_assistant.domain.models import (
    Chunk,
    Citation,
    HandoffReason,
    LoadedDocument,
    Message,
    Outcome,
    Page,
    Reply,
    Role,
    ScoredChunk,
    Ticket,
    TicketMessage,
    TicketStatus,
)

__all__ = [
    "Chunk",
    "Citation",
    "HandoffReason",
    "LoadedDocument",
    "Message",
    "Outcome",
    "Page",
    "Reply",
    "Role",
    "ScoredChunk",
    "Ticket",
    "TicketMessage",
    "TicketStatus",
]
