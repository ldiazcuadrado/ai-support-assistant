from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from support_assistant.errors import IngestError
from support_assistant.ingestion.loaders import (
    MAX_PAGE_BYTES,
    FetchedPage,
    discover_files,
    http_fetcher,
    load_file,
    load_page,
    parse_html,
    parse_markdown,
    sitemap_urls,
)
from tests.conftest import make_pdf


def test_markdown_title_from_first_heading() -> None:
    doc = parse_markdown("a.md", "# Opening hours\n\nWe open at 8.", fallback_title="A")
    assert doc.title == "Opening hours"
    assert "We open at 8." in doc.pages[0].text


def test_markdown_front_matter_title_wins_and_is_stripped() -> None:
    text = "---\ntitle: 'Real title'\ntags: x\n---\n# Heading\n\nBody"
    doc = parse_markdown("a.md", text, fallback_title="A")
    assert doc.title == "Real title"
    assert "tags" not in doc.pages[0].text


def test_markdown_without_heading_uses_file_name(tmp_path: Path) -> None:
    (tmp_path / "return-policy.md").write_text("Plain body")
    assert load_file(tmp_path / "return-policy.md").title == "Return policy"


def test_html_drops_scripts_navigation_and_keeps_title() -> None:
    html = (
        "<html><head><title> My  Page </title><script>var x=1</script></head>"
        "<body><nav>menu</nav><h1>Hello</h1><p>Body &amp; soul</p></body></html>"
    )
    doc = parse_html("p.html", html)
    text = doc.pages[0].text
    assert doc.title == "My Page"
    assert "Body & soul" in text
    assert "var x" not in text
    assert "menu" not in text


def test_pdf_pages_are_numbered(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    path.write_bytes(make_pdf(["Chain lubrication guide", "Brake bleeding guide"]))
    doc = load_file(path)
    assert [p.number for p in doc.pages] == [1, 2]
    assert "Brake bleeding" in doc.pages[1].text
    assert doc.title == "Manual"


def test_invalid_pdf_raises_ingest_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(IngestError):
        load_file(path)


def test_empty_file_raises_ingest_error(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("   \n")
    with pytest.raises(IngestError, match="no extractable text"):
        load_file(tmp_path / "empty.txt")


def test_unsupported_suffix_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text("a,b")
    with pytest.raises(IngestError, match="unsupported"):
        load_file(tmp_path / "data.csv")


def test_discover_files_filters_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / ".git").mkdir()
    for name in ["a.md", "sub/b.txt", "c.csv", ".git/d.md"]:
        (tmp_path / name).write_text("x")
    found = [p.relative_to(tmp_path).as_posix() for p in discover_files(tmp_path)]
    assert found == ["a.md", "sub/b.txt"]


def test_discover_files_missing_path(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="does not exist"):
        discover_files(tmp_path / "nope")


def test_source_is_relative_to_ingest_root(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("# T\n\nbody")
    assert load_file(tmp_path / "sub" / "a.md", tmp_path).source == "sub/a.md"


def test_load_page_dispatches_on_content_type() -> None:
    html = load_page(FetchedPage("https://x.test/a", "text/html", "<title>T</title><p>hi</p>"))
    plain = load_page(FetchedPage("https://x.test/b", "text/plain", "hello"))
    assert (html.title, plain.title) == ("T", "https://x.test/b")
    with pytest.raises(IngestError, match="content type"):
        load_page(FetchedPage("https://x.test/c", "application/zip", ""))


SITEMAP = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://x.test/a</loc></url>
  <url><loc> https://x.test/b </loc></url>
</urlset>"""

INDEX = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://x.test/s1.xml</loc></sitemap>
</sitemapindex>"""


def test_sitemap_urls_and_nested_index() -> None:
    pages = {"https://x.test/s1.xml": SITEMAP, "https://x.test/index.xml": INDEX}

    def fetch(url: str) -> FetchedPage:
        return FetchedPage(url, "application/xml", pages[url])

    assert sitemap_urls("https://x.test/s1.xml", fetch) == ["https://x.test/a", "https://x.test/b"]
    assert sitemap_urls("https://x.test/index.xml", fetch) == [
        "https://x.test/a",
        "https://x.test/b",
    ]
    assert sitemap_urls("https://x.test/s1.xml", fetch, limit=1) == ["https://x.test/a"]


def _serve(body: str) -> Callable[[str], FetchedPage]:
    return lambda url: FetchedPage(url, "text/xml", body)


def test_sitemap_rejects_invalid_xml_and_entity_bombs() -> None:
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><urlset>&a;</urlset>'
    for body in ("not xml", bomb):
        with pytest.raises(IngestError, match="invalid sitemap"):
            sitemap_urls("https://x.test/s.xml", _serve(body))


def test_http_fetcher_reads_pages_and_reports_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ok":
            return httpx.Response(
                200, text="<p>hi</p>", headers={"content-type": "text/html; charset=utf-8"}
            )
        if request.url.path == "/big":
            return httpx.Response(200, content=b"x" * (MAX_PAGE_BYTES + 1))
        return httpx.Response(404)

    fetch = http_fetcher(httpx.Client(transport=httpx.MockTransport(handler)))
    page = fetch("https://x.test/ok")
    assert (page.content_type, page.text) == ("text/html", "<p>hi</p>")
    for path in ("/missing", "/big"):
        with pytest.raises(IngestError):
            fetch(f"https://x.test{path}")
