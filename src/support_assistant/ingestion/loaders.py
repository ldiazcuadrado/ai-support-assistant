from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from support_assistant.domain import LoadedDocument, Page
from support_assistant.errors import IngestError

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".pdf", ".html", ".htm"})
MAX_PAGE_BYTES = 5_000_000

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FRONT_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)
_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True, slots=True)
class FetchedPage:
    url: str
    content_type: str
    text: str


Fetcher = Callable[[str], FetchedPage]


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise IngestError(f"path does not exist: {path}")
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )


def load_file(path: Path, root: Path | None = None) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(f"unsupported file type: {path.name}")
    base = root if root is not None and root.is_dir() else path.parent
    source = path.relative_to(base).as_posix()
    if suffix == ".pdf":
        return load_pdf(path, source)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return parse_html(source, text)
    if suffix == ".txt":
        return parse_text(source, text, fallback_title=_humanize(path.stem))
    return parse_markdown(source, text, fallback_title=_humanize(path.stem))


def parse_markdown(source: str, text: str, fallback_title: str) -> LoadedDocument:
    title = None
    front = _FRONT_MATTER.match(text)
    if front:
        found = _FRONT_TITLE.search(front.group(1))
        title = found.group(1) if found else None
        text = text[front.end() :]
    text = _COMMENT.sub("", text)
    if title is None:
        heading = _H1.search(text)
        title = heading.group(1) if heading else fallback_title
    return _document(source, title, [Page(text)])


def parse_text(source: str, text: str, fallback_title: str) -> LoadedDocument:
    return _document(source, fallback_title, [Page(text)])


def load_pdf(path: Path, source: str) -> LoadedDocument:
    try:
        reader = PdfReader(path)
        pages = [
            Page(page.extract_text() or "", number=i) for i, page in enumerate(reader.pages, 1)
        ]
        meta_title = reader.metadata.title if reader.metadata else None
    except (PyPdfError, OSError, ValueError) as exc:
        raise IngestError(f"cannot read PDF {path.name}: {exc}") from exc
    return _document(source, meta_title or _humanize(path.stem), pages)


class _TextExtractor(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "svg", "nav", "header", "footer", "aside"})
    _BLOCK = frozenset(
        {"p", "div", "section", "article", "main", "table", "ul", "ol", "blockquote", "pre"}
        | {f"h{n}" for n in range(1, 7)}
    )
    _LINE = frozenset({"li", "br", "tr", "dt", "dd"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n\n")
        elif tag in self._LINE:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def parse_html(source: str, html: str) -> LoadedDocument:
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    title = " ".join(extractor.title.split()) or source
    return _document(source, title, [Page("".join(extractor.parts))])


def load_page(page: FetchedPage) -> LoadedDocument:
    if "html" in page.content_type or not page.content_type:
        return parse_html(page.url, page.text)
    if page.content_type.startswith("text/"):
        return parse_text(page.url, page.text, fallback_title=page.url)
    raise IngestError(f"unsupported content type for {page.url}: {page.content_type}")


def http_fetcher(client: httpx.Client | None = None, timeout: float = 15.0) -> Fetcher:
    client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "support-assistant-ingest/0.1"},
    )

    def fetch(url: str) -> FetchedPage:
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise IngestError(f"cannot fetch {url}: {exc}") from exc
        if len(response.content) > MAX_PAGE_BYTES:
            raise IngestError(f"{url} is larger than {MAX_PAGE_BYTES} bytes")
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        return FetchedPage(url=url, content_type=content_type, text=response.text)

    return fetch


def sitemap_urls(url: str, fetch: Fetcher, limit: int = 200, _depth: int = 0) -> list[str]:
    """Return page URLs listed in a sitemap, following nested sitemap indexes."""
    page = fetch(url)
    try:
        root = ElementTree.fromstring(page.text.encode())
    except (ElementTree.ParseError, DefusedXmlException) as exc:
        raise IngestError(f"invalid sitemap {url}: {exc}") from exc
    locs = [
        (el.text or "").strip()
        for el in root.iter()
        if el.tag.rsplit("}", 1)[-1] == "loc" and (el.text or "").strip()
    ]
    if root.tag.rsplit("}", 1)[-1] != "sitemapindex":
        return locs[:limit]
    if _depth >= 2:
        return []
    urls: list[str] = []
    for child in locs:
        urls.extend(sitemap_urls(child, fetch, limit - len(urls), _depth + 1))
        if len(urls) >= limit:
            break
    return urls[:limit]


def _document(source: str, title: str, pages: list[Page]) -> LoadedDocument:
    kept = tuple(p for p in pages if p.text.strip())
    if not kept:
        raise IngestError(f"no extractable text in {source}")
    return LoadedDocument(source=source, title=" ".join(title.split()), pages=kept)


def _humanize(stem: str) -> str:
    return re.sub(r"[-_]+", " ", stem).strip().capitalize()
