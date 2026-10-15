from __future__ import annotations

import asyncio
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from support_assistant.domain import Ticket
from support_assistant.notify.base import format_ticket


@dataclass(frozen=True, slots=True)
class SmtpConfig:
    host: str
    port: int
    sender: str
    recipient: str
    username: str | None = None
    password: str | None = None
    starttls: bool = True


class EmailNotifier:
    def __init__(self, config: SmtpConfig) -> None:
        self._config = config

    async def notify_handoff(self, ticket: Ticket) -> None:
        await asyncio.to_thread(self._send, ticket)

    def _send(self, ticket: Ticket) -> None:
        cfg = self._config
        message = EmailMessage()
        message["Subject"] = f"[Support] Ticket #{ticket.id}: human handoff"
        message["From"] = cfg.sender
        message["To"] = cfg.recipient
        message.set_content(format_ticket(ticket))
        with smtplib.SMTP(cfg.host, cfg.port, timeout=15) as smtp:
            if cfg.starttls:
                smtp.starttls()
            if cfg.username and cfg.password:
                smtp.login(cfg.username, cfg.password)
            smtp.send_message(message)
