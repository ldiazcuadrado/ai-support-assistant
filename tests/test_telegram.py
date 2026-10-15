from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler

from support_assistant.channels.telegram import (
    TelegramChannel,
    build_application,
    format_reply,
    split_message,
)
from support_assistant.domain import Citation, Outcome, Reply
from support_assistant.factory import Container
from support_assistant.notify import TelegramDelivery, TelegramNotifier
from support_assistant.service.assistant import ERROR_TEXT

QUESTION = "What does a puncture repair cost?"


def _update(text: str | None, chat_id: int = 42) -> Any:
    update = MagicMock(spec=Update)
    update.effective_chat = MagicMock(id=chat_id, send_action=AsyncMock())
    update.effective_message = MagicMock(text=text, reply_text=AsyncMock())
    return update


@pytest.fixture
def channel(container: Container) -> TelegramChannel:
    return TelegramChannel(container.assistant, "Spoke & Chain")


def _sent(update: Any) -> list[str]:
    return [call.args[0] for call in update.effective_message.reply_text.await_args_list]


def test_format_reply_appends_sources() -> None:
    reply = Reply(
        "Fifteen. [1][2]",
        Outcome.ANSWERED,
        (Citation(1, "Pricing", "p.md"), Citation(2, "Manual", "m.pdf", page=3)),
    )
    assert format_reply(reply) == "Fifteen. [1][2]\n\nSources:\n[1] Pricing\n[2] Manual, page 3"
    assert format_reply(Reply("Hi", Outcome.HANDOFF)) == "Hi"


def test_split_message_respects_limit_and_boundaries() -> None:
    text = "\n\n".join(["word " * 40] * 10)
    parts = split_message(text, limit=500)
    assert all(len(p) <= 500 for p in parts)
    assert " ".join(" ".join(parts).split()) == " ".join(text.split())
    assert split_message("x" * 25, limit=10) == ["x" * 10, "x" * 10, "x" * 5]
    assert split_message("short") == ["short"]
    assert split_message("  ") == []


async def test_message_is_answered_with_sources(channel: TelegramChannel) -> None:
    update = _update(QUESTION)
    await channel.on_message(update, MagicMock())
    update.effective_chat.send_action.assert_awaited_once_with("typing")
    (message,) = _sent(update)
    assert "Sources:" in message and "[1]" in message


async def test_messages_without_text_are_ignored(channel: TelegramChannel) -> None:
    update = _update(None)
    await channel.on_message(update, MagicMock())
    update.effective_message.reply_text.assert_not_awaited()
    update.effective_message = None
    await channel.on_message(update, MagicMock())


async def test_human_command_opens_a_ticket(channel: TelegramChannel, container: Container) -> None:
    update = _update("/human", chat_id=7)
    await channel.on_human(update, MagicMock())
    assert "passed your conversation to our team" in _sent(update)[0]
    (ticket,) = container.tickets.list_tickets()
    assert ticket.session_id == "telegram:7"
    assert ticket.channel == "telegram"


async def test_human_phrase_in_text_is_handled_by_the_assistant(
    channel: TelegramChannel, container: Container
) -> None:
    update = _update("please let me talk to a human")
    await channel.on_message(update, MagicMock())
    assert len(container.tickets.list_tickets()) == 1


async def test_start_and_help(channel: TelegramChannel) -> None:
    for handler in (channel.on_start, channel.on_help):
        update = _update("/start")
        await handler(update, MagicMock())
        assert len(_sent(update)) == 1
    empty = _update(None)
    empty.effective_message = None
    await channel.on_start(empty, MagicMock())
    await channel.on_help(empty, MagicMock())


async def test_oversized_message_gets_a_friendly_reply(channel: TelegramChannel) -> None:
    assert await channel.answer(1, "x" * 5000) == [
        "Sorry, I cannot process that message: message is longer than 1000 characters."
    ]


async def test_rate_limited_user_is_told_to_wait(make_container: Callable[..., Container]) -> None:
    channel = TelegramChannel(make_container(rate_limit_requests=1).assistant, "x")
    await channel.answer(1, "human")
    (limited,) = await channel.answer(1, "human")
    assert "too quickly" in limited


async def test_request_human_rejects_bad_input(
    make_container: Callable[..., Container],
) -> None:
    channel = TelegramChannel(make_container(max_input_chars=20).assistant, "x")
    (message,) = await channel.request_human(1, "y" * 100)
    assert "cannot process" in message


async def test_error_handler_notifies_the_user(channel: TelegramChannel) -> None:
    update = _update("hi")
    context = MagicMock(error=RuntimeError("boom"))
    await channel.on_error(update, context)
    assert _sent(update) == [ERROR_TEXT]
    await channel.on_error(object(), context)


def test_application_registers_private_chat_handlers(channel: TelegramChannel) -> None:
    app = build_application("123456:TEST-TOKEN", channel)
    handlers = app.handlers[0]
    commands = [h for h in handlers if isinstance(h, CommandHandler)]
    assert {next(iter(h.commands)) for h in commands} == {"start", "help", "human"}
    assert any(isinstance(h, MessageHandler) for h in handlers)


async def test_telegram_notifier_and_delivery_use_the_sender(container: Container) -> None:
    sent: list[tuple[int | str, str]] = []

    async def sender(chat_id: int | str, text: str) -> None:
        sent.append((chat_id, text))

    ticket = container.tickets.create("telegram:55", "telegram", _reason(), "Where is my bike?")
    await TelegramNotifier(sender, "-1001").notify_handoff(ticket)
    assert sent[0][0] == "-1001"
    assert "ticket #1" in sent[0][1] and "Where is my bike?" in sent[0][1]

    delivery = TelegramDelivery(sender)
    assert await delivery.deliver(ticket, "It is ready.") is True
    assert sent[1] == ("55", "It is ready.")
    web_ticket = container.tickets.create("web:abc", "web", _reason(), "q")
    assert await delivery.deliver(web_ticket, "x") is False


def _reason() -> Any:
    from support_assistant.domain import HandoffReason

    return HandoffReason.USER_REQUEST
