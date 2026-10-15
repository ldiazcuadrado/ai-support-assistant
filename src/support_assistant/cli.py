from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from support_assistant import __version__
from support_assistant.config import Settings
from support_assistant.domain import TicketStatus
from support_assistant.errors import ConfigError, SupportAssistantError
from support_assistant.factory import Container, build_container, create_embedder
from support_assistant.ingestion import IngestPipeline, IngestReport, http_fetcher
from support_assistant.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-assistant", description="Documentation-grounded customer support assistant."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="load documents into the knowledge base")
    ingest.add_argument("path", nargs="?", type=Path, help="file or directory (md, txt, pdf, html)")
    ingest.add_argument(
        "--url", action="append", default=[], help="web page to ingest (repeatable)"
    )
    ingest.add_argument("--sitemap", help="sitemap URL whose pages are ingested")
    ingest.add_argument("--limit", type=int, default=200, help="maximum pages taken from a sitemap")
    ingest.add_argument("--reset", action="store_true", help="delete the existing knowledge first")

    sub.add_parser("sources", help="list ingested sources")

    serve = sub.add_parser("serve", help="run the HTTP API and web widget")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    sub.add_parser("telegram", help="run the Telegram bot")

    ask = sub.add_parser("ask", help="ask a question from the terminal")
    ask.add_argument("question")
    ask.add_argument("--session", default="cli")

    stats = sub.add_parser("stats", help="print analytics as JSON")
    stats.add_argument("--days", type=int, default=30)

    tickets = sub.add_parser("tickets", help="work with handoff tickets").add_subparsers(
        dest="action", required=True
    )
    listing = tickets.add_parser("list", help="list tickets")
    listing.add_argument("--status", choices=[s.value for s in TicketStatus])
    show = tickets.add_parser("show", help="show one ticket with its messages")
    show.add_argument("ticket_id", type=int)
    reply = tickets.add_parser("reply", help="reply to the customer of a ticket")
    reply.add_argument("ticket_id", type=int)
    reply.add_argument("message")
    close = tickets.add_parser("close", help="close a ticket")
    close.add_argument("ticket_id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_format)
        return _dispatch(args, settings)
    except SupportAssistantError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ConfigError) else 1


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "ingest":
        return _ingest(args, settings)
    container = build_container(settings)
    match args.command:
        case "sources":
            for source, title, count in container.knowledge.list_sources():
                print(f"{count:4d}  {source}  ({title})")
        case "serve":
            _serve(container, args.host or settings.host, args.port or settings.port)
        case "telegram":
            _telegram(container)
        case "ask":
            reply = asyncio.run(container.assistant.reply(args.session, "cli", args.question))
            print(reply.text)
            for citation in reply.citations:
                print(f"  [{citation.index}] {citation.title} ({citation.source})")
        case "stats":
            print(json.dumps(asdict(container.analytics.report(args.days)), indent=2))
        case "tickets":
            return _tickets(args, container)
    return 0


def _ingest(args: argparse.Namespace, settings: Settings) -> int:
    if args.path is None and not args.url and not args.sitemap:
        raise ConfigError("give a path, --url or --sitemap")
    container = build_container(settings)
    if args.reset:
        container.knowledge.clear()
    pipeline = IngestPipeline(
        container.knowledge,
        settings.chunk_size,
        settings.chunk_overlap,
        container.embedder or create_embedder(settings),
    )
    report = IngestReport()
    if args.path is not None:
        _merge(report, pipeline.ingest_path(args.path))
    if args.url or args.sitemap:
        fetch = http_fetcher()
        if args.url:
            _merge(report, pipeline.ingest_urls(args.url, fetch))
        if args.sitemap:
            _merge(report, pipeline.ingest_sitemap(args.sitemap, fetch, args.limit))
    print(f"ingested {report.documents} documents, {report.chunks} chunks")
    for source, reason in report.skipped:
        print(f"skipped {source}: {reason}", file=sys.stderr)
    return 0 if report.documents else 1


def _merge(total: IngestReport, part: IngestReport) -> None:
    total.documents += part.documents
    total.chunks += part.chunks
    total.skipped.extend(part.skipped)


def _serve(container: Container, host: str, port: int) -> None:  # pragma: no cover
    import uvicorn

    from support_assistant.channels.api import create_app

    _ = container.assistant  # fail fast on missing credentials
    uvicorn.run(create_app(container), host=host, port=port, log_config=None)


def _telegram(container: Container) -> None:
    from support_assistant.channels.telegram import TelegramChannel, build_application

    token = container.settings.telegram_bot_token
    if token is None:
        raise ConfigError("TELEGRAM_BOT_TOKEN is required to run the Telegram bot")
    channel = TelegramChannel(container.assistant, container.settings.business_name)
    build_application(token.get_secret_value(), channel).run_polling()


def _tickets(args: argparse.Namespace, container: Container) -> int:
    match args.action:
        case "list":
            status = TicketStatus(args.status) if args.status else None
            for t in container.tickets.list_tickets(status):
                print(
                    f"#{t.id}  {t.status.value:8}  {t.reason.value:18}  {t.session_id}  "
                    f"{t.question[:60]}"
                )
        case "show":
            ticket = container.tickets.get(args.ticket_id)
            if ticket is None:
                print(f"error: ticket {args.ticket_id} not found", file=sys.stderr)
                return 1
            print(
                f"#{ticket.id} [{ticket.status.value}] {ticket.reason.value} "
                f"via {ticket.session_id}"
            )
            for m in ticket.messages:
                print(f"  {m.created_at}  {m.author}: {m.content}")
        case "reply":
            _, delivered = asyncio.run(container.handoff.reply(args.ticket_id, args.message))
            print(
                "sent to customer"
                if delivered
                else "stored; the customer will see it in the web widget"
            )
        case "close":
            asyncio.run(container.handoff.close(args.ticket_id))
            print(f"ticket {args.ticket_id} closed")
    return 0
