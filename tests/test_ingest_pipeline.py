from __future__ import annotations

from pathlib import Path

from support_assistant.factory import Container
from support_assistant.ingestion import FetchedPage, IngestPipeline
from tests.conftest import KNOWLEDGE_BASE, make_pdf


def _pipeline(c: Container) -> IngestPipeline:
    return IngestPipeline(c.knowledge, c.settings.chunk_size, c.settings.chunk_overlap, c.embedder)


def test_sample_knowledge_base_is_ingested(container: Container) -> None:
    sources = {source for source, _, _ in container.knowledge.list_sources()}
    assert {"about.md", "ebike-service.html", "faq.txt", "services-and-pricing.md"} <= sources


def test_reingesting_replaces_instead_of_duplicating(container: Container) -> None:
    before = container.knowledge.count_chunks()
    _pipeline(container).ingest_path(KNOWLEDGE_BASE / "faq.txt")
    assert container.knowledge.count_chunks() == before


def test_reset_clears_everything(container: Container) -> None:
    container.knowledge.clear()
    assert container.knowledge.count_chunks() == 0


def test_failures_are_reported_and_do_not_stop_the_run(
    make_container: type, tmp_path: Path
) -> None:
    c = make_container()
    (tmp_path / "good.md").write_text("# Good\n\nUseful text")
    (tmp_path / "bad.pdf").write_bytes(b"garbage")
    (tmp_path / "blank.txt").write_text("")
    report = _pipeline(c).ingest_path(tmp_path)
    assert report.documents == 1
    assert {s.rsplit("/", 1)[-1] for s, _ in report.skipped} == {"bad.pdf", "blank.txt"}


def test_pdf_chunks_keep_page_numbers(make_container: type, tmp_path: Path) -> None:
    c = make_container()
    (tmp_path / "guide.pdf").write_bytes(make_pdf(["alpha text here", "bravo text here"]))
    _pipeline(c).ingest_path(tmp_path)
    hits = c.retriever.search("bravo", 3)
    assert [(h.chunk.source, h.chunk.page) for h in hits] == [("guide.pdf", 2)]


def test_urls_and_sitemap(make_container: type) -> None:
    c = make_container()
    pages = {
        "https://x.test/sitemap.xml": (
            "text/xml",
            "<urlset><url><loc>https://x.test/a</loc></url><url><loc>https://x.test/b</loc></url></urlset>",
        ),
        "https://x.test/a": ("text/html", "<title>Page A</title><p>Zebra crossing info</p>"),
        "https://x.test/b": ("application/zip", ""),
    }

    def fetch(url: str) -> FetchedPage:
        content_type, text = pages[url]
        return FetchedPage(url, content_type, text)

    report = _pipeline(c).ingest_sitemap("https://x.test/sitemap.xml", fetch)
    assert report.documents == 1
    assert len(report.skipped) == 1
    assert c.retriever.search("zebra", 1)[0].chunk.title == "Page A"
