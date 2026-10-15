from __future__ import annotations

import logging

from telegram import Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from support_assistant.domain import Reply
from support_assistant.errors import InputRejectedError, RateLimitedError
from support_assistant.service import Assistant
from support_assistant.service.assistant import ERROR_TEXT

logger = logging.getLogger(__name__)

CHANNEL = "telegram"
MESSAGE_LIMIT = 4096

WELCOME = (
    "Hello! Ask me anything about {business}. I answer from our documentation and cite my "
    "sources. Send /human at any time to talk to a person."
)
HELP = "Send a question in plain text. Use /human to reach a member of our team."


def format_reply(reply: Reply) -> str:
    if not reply.citations:
        return reply.text
    sources = "\n".join(
        f"[{c.index}] {c.title}" + (f", page {c.page}" if c.page is not None else "")
        for c in reply.citations
    )
    return f"{reply.text}\n\nSources:\n{sources}"


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split on paragraph, line or word boundaries so each part fits Telegram's limit."""
    parts: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        cut = max(
            rest.rfind("\n\n", 0, limit), rest.rfind("\n", 0, limit), rest.rfind(" ", 0, limit)
        )
        if cut <= 0:
            cut = limit
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


class TelegramChannel:
    def __init__(self, assistant: Assistant, business_name: str) -> None:
        self._assistant = assistant
        self._business = business_name

    async def answer(self, chat_id: int, text: str) -> list[str]:
        """Produce the message parts to send back for one customer message."""
        session_id = f"{CHANNEL}:{chat_id}"
        try:
            reply = await self._assistant.reply(session_id, CHANNEL, text)
        except RateLimitedError as exc:
            wait = max(1, round(exc.retry_after))
            return [f"You are sending messages too quickly. Please wait {wait}s."]
        except InputRejectedError as exc:
            return [f"Sorry, I cannot process that message: {exc}."]
        return split_message(format_reply(reply))

    async def request_human(self, chat_id: int, text: str) -> list[str]:
        session_id = f"{CHANNEL}:{chat_id}"
        try:
            reply = await self._assistant.request_human(session_id, CHANNEL, text)
        except InputRejectedError as exc:
            return [f"Sorry, I cannot process that message: {exc}."]
        return split_message(reply.text)

    async def on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if update.effective_message is not None:
            await update.effective_message.reply_text(WELCOME.format(business=self._business))

    async def on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if update.effective_message is not None:
            await update.effective_message.reply_text(HELP)

    async def on_human(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        message, chat = update.effective_message, update.effective_chat
        if message is None or chat is None:
            return
        await self._send(message, await self.request_human(chat.id, "Customer asked for a human."))

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        message, chat = update.effective_message, update.effective_chat
        if message is None or chat is None or not message.text:
            return
        await chat.send_action("typing")
        await self._send(message, await self.answer(chat.id, message.text))

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("telegram_handler_failed", exc_info=context.error)
        if isinstance(update, Update) and update.effective_message is not None:
            await update.effective_message.reply_text(ERROR_TEXT)

    @staticmethod
    async def _send(message: Message, parts: list[str]) -> None:
        for part in parts:
            await message.reply_text(part)


def build_application(token: str, channel: TelegramChannel) -> Application:  # type: ignore[type-arg]
    app = ApplicationBuilder().token(token).build()
    private = filters.ChatType.PRIVATE
    app.add_handler(CommandHandler("start", channel.on_start, filters=private))
    app.add_handler(CommandHandler("help", channel.on_help, filters=private))
    app.add_handler(CommandHandler("human", channel.on_human, filters=private))
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, channel.on_message))
    app.add_error_handler(channel.on_error)
    return app
