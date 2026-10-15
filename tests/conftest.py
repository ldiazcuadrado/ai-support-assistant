from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from support_assistant.config import Settings
from support_assistant.domain import Ticket
from support_assistant.factory import Container, build_container
from support_assistant.ingestion import IngestPipeline
from support_assistant.llm import FakeLLM
from support_assistant.notify import Notifier

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = ROOT / "knowledge_base"


class RecordingNotifier:
    def __init__(self) -> None:
        self.tickets: list[Ticket] = []

    async def notify_handoff(self, ticket: Ticket) -> None:
        self.tickets.append(ticket)


class RecordingDelivery:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def deliver(self, ticket: Ticket, text: str) -> bool:
        self.sent.append((ticket.id, text))
        return True


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page."""
    objects: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_path=tmp_path / "test.db",
        llm_provider="fake",
        admin_token="secret-token",
        rate_limit_requests=50,
        log_format="console",
    )


@pytest.fixture
def make_container(settings: Settings) -> Callable[..., Container]:
    def factory(
        llm: FakeLLM | None = None, *, notifier: Notifier | None = None, **overrides: Any
    ) -> Container:
        cfg = settings.model_copy(update=overrides)
        return build_container(cfg, llm=llm or FakeLLM(), notifier=notifier)

    return factory


@pytest.fixture
def container(make_container: Callable[..., Container]) -> Container:
    c = make_container()
    pipeline = IngestPipeline(c.knowledge, c.settings.chunk_size, c.settings.chunk_overlap)
    report = pipeline.ingest_path(KNOWLEDGE_BASE)
    assert report.documents >= 6
    return c
