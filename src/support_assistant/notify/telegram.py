from __future__ import annotations

from collections.abc import Awaitable, Callable

from telegram import Bot

from support_assistant.domain import Ticket
from support_assistant.notify.base import format_ticket

TelegramSender = Callable[[int | str, str], Awaitable[None]]
SESSION_PREFIX = "telegram:"


def bot_sender(token: str) -> TelegramSender:  # pragma: no cover - performs network I/O
    async def send(chat_id: int | str, text: str) -> None:
        async with Bot(token) as bot:
            await bot.send_message(chat_id=chat_id, text=text)

    return send


class TelegramNotifier:
    def __init__(self, send: TelegramSender, chat_id: str) -> None:
        self._send = send
        self._chat_id = chat_id

    async def notify_handoff(self, ticket: Ticket) -> None:
        await self._send(self._chat_id, format_ticket(ticket))


class TelegramDelivery:
    """Sends operator replies to customers whose session started on Telegram."""

    def __init__(self, send: TelegramSender) -> None:
        self._send = send

    async def deliver(self, ticket: Ticket, text: str) -> bool:
        if not ticket.session_id.startswith(SESSION_PREFIX):
            return False
        await self._send(ticket.session_id.removeprefix(SESSION_PREFIX), text)
        return True
