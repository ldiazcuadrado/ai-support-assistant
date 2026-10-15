from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Outcome(StrEnum):
    ANSWERED = "answered"
    UNANSWERED = "unanswered"
    HANDOFF = "handoff"
    ERROR = "error"


class HandoffReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    USER_REQUEST = "user_request"
    REPEATED_FAILURES = "repeated_failures"


class TicketStatus(StrEnum):
    OPEN = "open"
    ANSWERED = "answered"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Page:
    """A unit of extracted text. ``number`` is only set for paginated formats."""

    text: str
    number: int | None = None


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    source: str
    title: str
    pages: tuple[Page, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    source: str
    title: str
    page: int | None
    position: int
    text: str
    id: int | None = None


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk.

    ``score`` is only meaningful for ordering within one result list.
    ``relevance`` is normalised to [0, 1] and is what confidence checks use.
    """

    chunk: Chunk
    score: float
    relevance: float


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Citation:
    index: int
    title: str
    source: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    outcome: Outcome
    citations: tuple[Citation, ...] = ()
    ticket_id: int | None = None
    handoff_reason: HandoffReason | None = None


@dataclass(frozen=True, slots=True)
class TicketMessage:
    id: int
    ticket_id: int
    author: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class Ticket:
    id: int
    session_id: str
    channel: str
    status: TicketStatus
    reason: HandoffReason
    question: str
    created_at: str
    updated_at: str
    messages: tuple[TicketMessage, ...] = field(default_factory=tuple)
