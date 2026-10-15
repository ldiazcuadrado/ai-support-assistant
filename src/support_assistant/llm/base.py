from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from support_assistant.domain import Message, ScoredChunk


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A fully rendered model call.

    ``messages`` already contains the retrieved context. ``sources`` repeats the retrieved chunks
    for clients that do not read the prompt, such as the offline fake.
    """

    system: str
    messages: tuple[Message, ...]
    max_tokens: int
    sources: tuple[ScoredChunk, ...] = ()


class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> str:
        """Return the full response. Raises ``LLMError`` on failure."""
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield the response incrementally. Raises ``LLMError`` on failure."""
        ...
