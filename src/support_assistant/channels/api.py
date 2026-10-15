from __future__ import annotations

import json
import logging
import secrets
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from support_assistant import __version__
from support_assistant.domain import Reply, Ticket, TicketStatus
from support_assistant.errors import InputRejectedError, RateLimitedError, TicketNotFoundError
from support_assistant.factory import Container
from support_assistant.service import DoneEvent, StreamEvent, TokenEvent

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "web" / "static"
SESSION_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"
CHANNEL = "web"


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_PATTERN)
    message: str = Field(min_length=1, max_length=20_000)


class CitationOut(BaseModel):
    index: int
    title: str
    source: str
    page: int | None


class ChatResponse(BaseModel):
    answer: str
    outcome: str
    citations: list[CitationOut]
    ticket_id: int | None


class OperatorReplyOut(BaseModel):
    id: int
    text: str
    created_at: str


class OperatorReplyIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def _chat_response(reply: Reply) -> ChatResponse:
    return ChatResponse(
        answer=reply.text,
        outcome=reply.outcome.value,
        citations=[CitationOut(**asdict(c)) for c in reply.citations],
        ticket_id=reply.ticket_id,
    )


def _ticket_json(ticket: Ticket) -> dict[str, Any]:
    return {
        "id": ticket.id,
        "session_id": ticket.session_id,
        "channel": ticket.channel,
        "status": ticket.status.value,
        "reason": ticket.reason.value,
        "question": ticket.question,
        "created_at": ticket.created_at,
        "updated_at": ticket.updated_at,
        "messages": [asdict(m) for m in ticket.messages],
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(container: Container) -> FastAPI:
    settings = container.settings
    app = FastAPI(title="AI Support Assistant", version=__version__, docs_url=None, redoc_url=None)
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.exception_handler(InputRejectedError)
    async def _rejected(_: Request, exc: InputRejectedError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(RateLimitedError)
    async def _limited(_: Request, exc: RateLimitedError) -> JSONResponse:
        return JSONResponse(
            {"detail": "Too many requests. Please slow down."},
            status_code=429,
            headers={"Retry-After": str(max(1, round(exc.retry_after)))},
        )

    @app.exception_handler(TicketNotFoundError)
    async def _missing(_: Request, exc: TicketNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    def limit_by_ip(request: Request) -> None:
        container.ip_limiter.check(request.client.host if request.client else "unknown")

    ip_limit = Depends(limit_by_ip)

    def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
        token = settings.admin_token
        if token is None or not token.get_secret_value():
            raise HTTPException(503, "Admin API is disabled: set ADMIN_TOKEN")
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied.encode(), token.get_secret_value().encode()):
            raise HTTPException(401, "Invalid admin token", headers={"WWW-Authenticate": "Bearer"})

    admin = Depends(require_admin)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "chunks": container.knowledge.count_chunks(),
        }

    @app.post("/api/chat", dependencies=[ip_limit])
    async def chat(body: ChatRequest) -> ChatResponse:
        reply = await container.assistant.reply(
            f"{CHANNEL}:{body.session_id}", CHANNEL, body.message
        )
        return _chat_response(reply)

    @app.post("/api/chat/stream", dependencies=[ip_limit])
    async def chat_stream(body: ChatRequest) -> StreamingResponse:
        events = container.assistant.stream(f"{CHANNEL}:{body.session_id}", CHANNEL, body.message)
        first = await anext(events)  # surfaces rejected input and rate limits as HTTP errors

        async def body_iter() -> AsyncIterator[str]:
            try:
                yield _encode(first)
                async for event in events:
                    yield _encode(event)
            except Exception:
                logger.exception("stream_failed")
                yield _sse("error", {"detail": "The answer could not be completed."})

        return StreamingResponse(
            body_iter(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{session_id}/operator-replies", dependencies=[ip_limit])
    def operator_replies(
        session_id: Annotated[str, PathParam(pattern=SESSION_PATTERN)],
        after: Annotated[int, Query(ge=0)] = 0,
    ) -> list[OperatorReplyOut]:
        replies = container.tickets.operator_replies(f"{CHANNEL}:{session_id}", after)
        return [OperatorReplyOut(id=r.id, text=r.content, created_at=r.created_at) for r in replies]

    @app.get("/admin/tickets", dependencies=[admin])
    def list_tickets(
        status: TicketStatus | None = None, limit: int = Query(100, ge=1, le=500)
    ) -> Any:
        return [_ticket_json(t) for t in container.tickets.list_tickets(status, limit)]

    @app.get("/admin/tickets/{ticket_id}", dependencies=[admin])
    def get_ticket(ticket_id: int) -> dict[str, Any]:
        ticket = container.tickets.get(ticket_id)
        if ticket is None:
            raise TicketNotFoundError(f"ticket {ticket_id} not found")
        return _ticket_json(ticket)

    @app.post("/admin/tickets/{ticket_id}/reply", dependencies=[admin])
    async def reply_to_ticket(ticket_id: int, body: OperatorReplyIn) -> dict[str, Any]:
        message, delivered = await container.handoff.reply(ticket_id, body.message)
        return {"message_id": message.id, "delivered": delivered}

    @app.post("/admin/tickets/{ticket_id}/close", dependencies=[admin])
    async def close_ticket(ticket_id: int) -> dict[str, str]:
        await container.handoff.close(ticket_id)
        return {"status": "closed"}

    @app.get("/admin/sessions/{session_id}/messages", dependencies=[admin])
    def session_messages(session_id: str) -> list[dict[str, str]]:
        return [
            {"role": role, "content": content, "created_at": at}
            for role, content, at in container.conversations.transcript(session_id)
        ]

    @app.get("/admin/analytics", dependencies=[admin])
    def analytics(days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
        return asdict(container.analytics.report(days))

    def static(name: str, media_type: str) -> Response:
        return FileResponse(
            STATIC_DIR / name,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=300"},
        )

    @app.get("/widget.js", include_in_schema=False)
    def widget_js() -> Response:
        return static("widget.js", "text/javascript")

    @app.get("/widget.css", include_in_schema=False)
    def widget_css() -> Response:
        return static("widget.css", "text/css")

    @app.get("/", include_in_schema=False)
    def demo_page() -> Response:
        return static("index.html", "text/html")

    return app


def _encode(event: StreamEvent) -> str:
    if isinstance(event, TokenEvent):
        return _sse("token", {"text": event.text})
    assert isinstance(event, DoneEvent)
    return _sse("done", _chat_response(event.reply).model_dump())
