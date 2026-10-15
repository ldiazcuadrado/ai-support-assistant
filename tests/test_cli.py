from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from support_assistant import cli
from tests.conftest import KNOWLEDGE_BASE


@pytest.fixture(autouse=True)
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LOG_FORMAT", "console")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)


def _ingest() -> None:
    assert cli.main(["ingest", str(KNOWLEDGE_BASE)]) == 0


def test_ingest_sources_and_ask(capsys: pytest.CaptureFixture[str]) -> None:
    _ingest()
    assert "ingested 7 documents" in capsys.readouterr().out
    assert cli.main(["sources"]) == 0
    assert "faq.txt" in capsys.readouterr().out
    assert cli.main(["ask", "What does a puncture repair cost?"]) == 0
    out = capsys.readouterr().out
    assert "[1]" in out and "(services-and-pricing.md)" in out


def test_ingest_reset_and_partial_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ingest()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ok.md").write_text("# Only\n\ntext about warranty")
    (docs / "bad.pdf").write_bytes(b"nope")
    assert cli.main(["ingest", str(docs), "--reset"]) == 0
    captured = capsys.readouterr()
    assert "skipped" in captured.err and "bad.pdf" in captured.err
    cli.main(["sources"])
    assert capsys.readouterr().out.count("\n") == 1


def test_ingest_requires_a_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ingest"]) == 2
    assert "give a path" in capsys.readouterr().err


def test_ingest_with_nothing_loaded_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["ingest", str(empty)]) == 1


def test_ingest_urls_and_sitemap_use_the_http_fetcher(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from support_assistant.ingestion import FetchedPage

    def fake_fetcher() -> Any:
        def fetch(url: str) -> FetchedPage:
            if url.endswith("sitemap.xml"):
                return FetchedPage(
                    url, "text/xml", "<urlset><url><loc>https://x.test/p</loc></url></urlset>"
                )
            return FetchedPage(url, "text/html", "<title>Remote</title><p>remote text</p>")

        return fetch

    monkeypatch.setattr(cli, "http_fetcher", fake_fetcher)
    assert (
        cli.main(["ingest", "--url", "https://x.test/a", "--sitemap", "https://x.test/sitemap.xml"])
        == 0
    )
    assert "ingested 2 documents" in capsys.readouterr().out


def test_missing_api_key_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["ask", "hello"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_ticket_commands(capsys: pytest.CaptureFixture[str]) -> None:
    _ingest()
    cli.main(["ask", "please connect me to a human"])
    capsys.readouterr()

    assert cli.main(["tickets", "list", "--status", "open"]) == 0
    assert "#1" in capsys.readouterr().out
    assert cli.main(["tickets", "show", "1"]) == 0
    assert "customer:" in capsys.readouterr().out
    assert cli.main(["tickets", "show", "99"]) == 1
    assert cli.main(["tickets", "reply", "1", "We will call you."]) == 0
    assert "stored" in capsys.readouterr().out
    assert cli.main(["tickets", "reply", "99", "x"]) == 1
    assert cli.main(["tickets", "close", "1"]) == 0
    assert cli.main(["tickets", "list"]) == 0
    assert "closed" in capsys.readouterr().out


def test_stats_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    _ingest()
    cli.main(["ask", "What does a puncture repair cost?"])
    capsys.readouterr()
    assert cli.main(["stats", "--days", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["answered"] == 1


def test_telegram_command_needs_a_token(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["telegram"]) == 2
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err


def test_telegram_command_starts_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    app = MagicMock()
    build = MagicMock(return_value=app)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setattr("support_assistant.channels.telegram.build_application", build)
    assert cli.main(["telegram"]) == 0
    assert build.call_args.args[0] == "123:abc"
    app.run_polling.assert_called_once()


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
