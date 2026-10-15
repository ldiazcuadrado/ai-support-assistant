from __future__ import annotations

import asyncio
import logging
import re

from support_assistant.domain import HandoffReason, Ticket, TicketMessage, TicketStatus
from support_assistant.errors import TicketNotFoundError
from support_assistant.notify import Notifier, ReplyDelivery
from support_assistant.storage import TicketStore

logger = logging.getLogger(__name__)

_HUMAN_REQUEST = re.compile(
    r"\b("
    r"human|real person|live (?:agent|person|chat)|operator|representative|manager|"
    r"(?:speak|talk|chat) (?:to|with) (?:someone|somebody|a person|an? agent|a member)|"
    r"customer (?:service|support) (?:agent|rep)|an agent|a person"
    r")\b",
    re.IGNORECASE,
)


def wants_human(text: str) -> bool:
    return _HUMAN_REQUEST.search(text) is not None


class HandoffService:
    """Creates tickets, notifies operators and carries operator replies back to customers."""

    def __init__(
        self,
        tickets: TicketStore,
        notifier: Notifier | None = None,
        delivery: ReplyDelivery | None = None,
    ) -> None:
        self._tickets = tickets
        self._notifier = notifier
        self._delivery = delivery

    async def open(
        self, session_id: str, channel: str, reason: HandoffReason, question: str
    ) -> tuple[Ticket, bool]:
        """Return ``(ticket, is_new)``. A session with a live ticket reuses it."""
        existing = await asyncio.to_thread(self._tickets.find_active, session_id)
        if existing is not None:
            await asyncio.to_thread(self._tickets.add_message, existing.id, "customer", question)
            reopened = existing.status is TicketStatus.ANSWERED
            if reopened:
                await asyncio.to_thread(self._tickets.set_status, existing.id, TicketStatus.OPEN)
                await self._notify(existing)
            return existing, False
        ticket = await asyncio.to_thread(
            self._tickets.create, session_id, channel, reason, question
        )
        logger.info("handoff_opened", extra={"ticket_id": ticket.id, "reason": reason.value})
        await self._notify(ticket)
        return ticket, True

    async def reply(self, ticket_id: int, text: str) -> tuple[TicketMessage, bool]:
        """Record an operator reply; returns the message and whether it reached the customer."""
        ticket = await asyncio.to_thread(self._tickets.get, ticket_id)
        if ticket is None:
            raise TicketNotFoundError(f"ticket {ticket_id} not found")
        message = await asyncio.to_thread(self._tickets.add_message, ticket_id, "operator", text)
        await asyncio.to_thread(self._tickets.set_status, ticket_id, TicketStatus.ANSWERED)
        delivered = False
        if self._delivery is not None:
            try:
                delivered = await self._delivery.deliver(ticket, text)
            except Exception:
                logger.exception("reply_delivery_failed", extra={"ticket_id": ticket_id})
        return message, delivered

    async def close(self, ticket_id: int) -> None:
        if await asyncio.to_thread(self._tickets.get, ticket_id) is None:
            raise TicketNotFoundError(f"ticket {ticket_id} not found")
        await asyncio.to_thread(self._tickets.set_status, ticket_id, TicketStatus.CLOSED)

    async def _notify(self, ticket: Ticket) -> None:
        if self._notifier is None:
            logger.warning("no_notifier_configured", extra={"ticket_id": ticket.id})
            return
        try:
            await self._notifier.notify_handoff(ticket)
        except Exception:
            logger.exception("notification_failed", extra={"ticket_id": ticket.id})
