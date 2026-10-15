from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from support_assistant.domain import (
    Citation,
    HandoffReason,
    Message,
    Outcome,
    Reply,
    Role,
    ScoredChunk,
)
from support_assistant.errors import LLMError
from support_assistant.llm import (
    NO_ANSWER,
    LLMClient,
    LLMRequest,
    build_system_prompt,
    render_user_message,
)
from support_assistant.retrieval import Retriever
from support_assistant.retrieval.text import tokens
from support_assistant.service.handoff import HandoffService, wants_human
from support_assistant.service.ratelimit import RateLimiter
from support_assistant.service.safety import sanitize_input
from support_assistant.storage import ConversationStore

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")

UNANSWERED_TEXT = (
    "I could not find the answer to that in our documentation. You can rephrase the question, "
    "or ask for a person and I will pass it to our team."
)
ERROR_TEXT = "Sorry, I am unable to answer right now. Please try again in a moment."
HANDOFF_TEXT = {
    HandoffReason.LOW_CONFIDENCE: (
        "I could not find this in our documentation, so I have passed your question to our team."
    ),
    HandoffReason.USER_REQUEST: "Of course. I have passed your conversation to our team.",
    HandoffReason.REPEATED_FAILURES: (
        "I have not been able to answer your questions, so I have passed this to our team."
    ),
}
HANDOFF_SUFFIX = " A person will reply here as soon as possible (ticket #{ticket_id})."
HANDOFF_EXISTING = (
    "Your request is already with our team (ticket #{ticket_id}). They will reply here."
)


@dataclass(frozen=True, slots=True)
class AssistantConfig:
    business_name: str
    top_k: int = 5
    min_relevance: float = 0.4
    memory_turns: int = 6
    max_input_chars: int = 1000
    max_output_tokens: int = 1024
    handoff_after_failures: int = 2


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class DoneEvent:
    reply: Reply


StreamEvent = TokenEvent | DoneEvent


@dataclass(frozen=True, slots=True)
class _Turn:
    session_id: str
    channel: str
    question: str
    top_relevance: float
    request: LLMRequest | None = None
    sources: tuple[ScoredChunk, ...] = ()
    immediate: Reply | None = None


class Assistant:
    """Answers customer questions from retrieved documentation and escalates to humans."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: LLMClient,
        conversations: ConversationStore,
        handoff: HandoffService,
        config: AssistantConfig,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._conversations = conversations
        self._handoff = handoff
        self._config = config
        self._limiter = limiter
        self._system = build_system_prompt(config.business_name)

    async def reply(self, session_id: str, channel: str, text: str) -> Reply:
        turn = await self._prepare(session_id, channel, text)
        if turn.immediate is not None or turn.request is None:
            return self._require_immediate(turn)
        try:
            raw = await self._llm.complete(turn.request)
        except LLMError:
            logger.exception("llm_failed", extra={"session_id": session_id})
            return await self._finish_failure(turn, Outcome.ERROR)
        return await self._finish(turn, raw)

    async def stream(self, session_id: str, channel: str, text: str) -> AsyncIterator[StreamEvent]:
        """Yield tokens as they arrive, then a final ``DoneEvent`` carrying the logged reply."""
        turn = await self._prepare(session_id, channel, text)
        if turn.immediate is not None or turn.request is None:
            reply = self._require_immediate(turn)
            yield TokenEvent(reply.text)
            yield DoneEvent(reply)
            return

        full = ""
        pending = ""
        passthrough = False
        try:
            async for piece in self._llm.stream(turn.request):
                full += piece
                if passthrough:
                    yield TokenEvent(piece)
                    continue
                pending += piece
                head = pending.lstrip()
                if NO_ANSWER.startswith(head) or head.startswith(NO_ANSWER):
                    continue
                passthrough = True
                yield TokenEvent(pending)
        except LLMError:
            logger.exception("llm_failed", extra={"session_id": session_id})
            reply = await self._finish_failure(turn, Outcome.ERROR)
            if not passthrough:
                yield TokenEvent(reply.text)
            yield DoneEvent(reply)
            return

        if not passthrough and pending and not pending.lstrip().startswith(NO_ANSWER):
            yield TokenEvent(pending)
        reply = await self._finish(turn, full)
        if not passthrough and reply.outcome is not Outcome.ANSWERED:
            yield TokenEvent(reply.text)
        yield DoneEvent(reply)

    async def request_human(self, session_id: str, channel: str, text: str) -> Reply:
        """Explicit handoff, used by channel commands such as Telegram's ``/human``."""
        question = sanitize_input(text, self._config.max_input_chars)
        return await self._handoff_reply(
            session_id, channel, question, HandoffReason.USER_REQUEST, None
        )

    async def _prepare(self, session_id: str, channel: str, text: str) -> _Turn:
        cfg = self._config
        if self._limiter is not None:
            self._limiter.check(session_id)
        question = sanitize_input(text, cfg.max_input_chars)

        if wants_human(question):
            reply = await self._handoff_reply(
                session_id, channel, question, HandoffReason.USER_REQUEST, None
            )
            return _Turn(session_id, channel, question, 0.0, immediate=reply)

        history = await asyncio.to_thread(self._conversations.recent, session_id, cfg.memory_turns)
        hits = await asyncio.to_thread(
            self._retriever.search, _retrieval_query(question, history), cfg.top_k
        )
        top = max((h.relevance for h in hits), default=0.0)
        sources = tuple(h for h in hits if h.relevance >= cfg.min_relevance)
        if not sources:
            reply = await self._handoff_reply(
                session_id, channel, question, HandoffReason.LOW_CONFIDENCE, top
            )
            return _Turn(session_id, channel, question, top, immediate=reply)

        messages = (*history, Message(Role.USER, render_user_message(question, sources)))
        request = LLMRequest(
            system=self._system,
            messages=messages,
            max_tokens=cfg.max_output_tokens,
            sources=sources,
        )
        return _Turn(session_id, channel, question, top, request=request, sources=sources)

    async def _finish(self, turn: _Turn, raw: str) -> Reply:
        text = raw.strip()
        if not text or text.startswith(NO_ANSWER):
            return await self._finish_failure(turn, Outcome.UNANSWERED)
        reply = Reply(text=text, outcome=Outcome.ANSWERED, citations=_citations(text, turn.sources))
        await self._log(turn, reply)
        return reply

    async def _finish_failure(self, turn: _Turn, outcome: Outcome) -> Reply:
        limit = self._config.handoff_after_failures
        previous = await asyncio.to_thread(
            self._conversations.consecutive_failures, turn.session_id, limit
        )
        if previous + 1 >= limit:
            return await self._handoff_reply(
                turn.session_id,
                turn.channel,
                turn.question,
                HandoffReason.REPEATED_FAILURES,
                turn.top_relevance,
            )
        text = UNANSWERED_TEXT if outcome is Outcome.UNANSWERED else ERROR_TEXT
        reply = Reply(text=text, outcome=outcome)
        await self._log(turn, reply)
        return reply

    async def _handoff_reply(
        self,
        session_id: str,
        channel: str,
        question: str,
        reason: HandoffReason,
        top_relevance: float | None,
    ) -> Reply:
        ticket, is_new = await self._handoff.open(session_id, channel, reason, question)
        if is_new:
            text = HANDOFF_TEXT[reason] + HANDOFF_SUFFIX.format(ticket_id=ticket.id)
        else:
            text = HANDOFF_EXISTING.format(ticket_id=ticket.id)
        reply = Reply(
            text=text, outcome=Outcome.HANDOFF, ticket_id=ticket.id, handoff_reason=reason
        )
        await asyncio.to_thread(
            self._conversations.add_exchange,
            session_id,
            channel,
            question,
            text,
            Outcome.HANDOFF,
            reason=reason.value,
            top_relevance=top_relevance,
        )
        return reply

    async def _log(self, turn: _Turn, reply: Reply) -> None:
        await asyncio.to_thread(
            self._conversations.add_exchange,
            turn.session_id,
            turn.channel,
            turn.question,
            reply.text,
            reply.outcome,
            top_relevance=turn.top_relevance,
        )
        logger.info(
            "exchange",
            extra={
                "session_id": turn.session_id,
                "outcome": reply.outcome.value,
                "top_relevance": round(turn.top_relevance, 3),
            },
        )

    @staticmethod
    def _require_immediate(turn: _Turn) -> Reply:
        assert turn.immediate is not None
        return turn.immediate


def _retrieval_query(question: str, history: list[Message]) -> str:
    """Short follow-ups ("and how much?") borrow the previous question for retrieval."""
    if len(tokens(question)) >= 3:
        return question
    previous = next((m.content for m in reversed(history) if m.role is Role.USER), "")
    return f"{previous} {question}".strip()


def _citations(text: str, sources: tuple[ScoredChunk, ...]) -> tuple[Citation, ...]:
    cited = sorted({int(n) for n in _CITATION.findall(text)})
    return tuple(
        Citation(
            index=n,
            title=sources[n - 1].chunk.title,
            source=sources[n - 1].chunk.source,
            page=sources[n - 1].chunk.page,
        )
        for n in cited
        if 1 <= n <= len(sources)
    )
