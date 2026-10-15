from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

from support_assistant.domain import Ticket

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    async def notify_handoff(self, ticket: Ticket) -> None:
        """Tell a human operator that a ticket needs attention."""
        ...


class ReplyDelivery(Protocol):
    async def deliver(self, ticket: Ticket, text: str) -> bool:
        """Push an operator reply to the customer. Returns False if the channel cannot."""
        ...


class CompositeNotifier:
    """Fans out to several notifiers; one failing never blocks the others."""

    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        self._notifiers = list(notifiers)

    async def notify_handoff(self, ticket: Ticket) -> None:
        results = await asyncio.gather(
            *(n.notify_handoff(ticket) for n in self._notifiers), return_exceptions=True
        )
        for notifier, result in zip(self._notifiers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "notifier_failed",
                    extra={"notifier": type(notifier).__name__, "ticket_id": ticket.id},
                    exc_info=result,
                )


def format_ticket(ticket: Ticket) -> str:
    question = ticket.question if len(ticket.question) <= 500 else ticket.question[:497] + "..."
    return (
        f"Support ticket #{ticket.id} needs a human ({ticket.reason.value})\n"
        f"Channel: {ticket.channel}\n"
        f"Session: {ticket.session_id}\n\n"
        f"Customer message:\n{question}\n\n"
        f'Reply with: support-assistant tickets reply {ticket.id} "..."'
    )
