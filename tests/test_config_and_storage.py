from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from support_assistant.config import Settings
from support_assistant.domain import Chunk, HandoffReason, Outcome, Role
from support_assistant.logging_config import JsonFormatter, configure_logging
from support_assistant.storage import ConversationStore, Database, KnowledgeStore, TicketStore


def test_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.test, https://b.test,")
    s = Settings(_env_file=None)
    assert s.anthropic_model == "claude-test"
    assert s.cors_origin_list == ["https://a.test", "https://b.test"]
    assert Settings(_env_file=None).business_name


def test_default_model_is_claude_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert Settings(_env_file=None).anthropic_model == "claude-sonnet-5-5"


def test_secrets_are_not_leaked_in_repr() -> None:
    s = Settings(_env_file=None, anthropic_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(s)


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)


def test_json_logging_includes_extras(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    try:
        logging.getLogger("t").info("hello", extra={"ticket_id": 3})
        try:
            raise ValueError("bad")
        except ValueError:
            logging.getLogger("t").exception("failed")
    finally:
        logging.getLogger().handlers.clear()
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert lines[0]["event"] == "hello" and lines[0]["ticket_id"] == 3
    assert lines[0]["level"] == "info"
    assert "ValueError" in lines[1]["exception"]


def test_console_logging_format(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", "console")
    try:
        logging.getLogger("t").debug("plain")
    finally:
        logging.getLogger().handlers.clear()
    assert "plain" in capsys.readouterr().err
    assert JsonFormatter().format(logging.makeLogRecord({"msg": "x"}))


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "nested" / "t.db")
    database.initialize()
    return database


def test_initialize_is_idempotent(db: Database) -> None:
    db.initialize()


def test_transaction_rolls_back_on_error(db: Database) -> None:
    with pytest.raises(RuntimeError), db.connection() as conn:
        conn.execute("INSERT INTO chunks(source,title,position,text) VALUES ('s','t',0,'x')")
        raise RuntimeError
    assert KnowledgeStore(db).count_chunks() == 0


def test_knowledge_store_replace_and_fts_stay_in_sync(db: Database) -> None:
    store = KnowledgeStore(db)
    store.replace_source("a.md", [Chunk("a.md", "A", None, 0, "alpha bravo")])
    store.replace_source("a.md", [Chunk("a.md", "A", None, 0, "charlie delta")])
    assert store.search_fts('"alpha"', 5) == []
    assert [c.text for c, _ in store.search_fts('"charlie"', 5)] == ["charlie delta"]
    assert store.get_many([]) == {}


def test_embeddings_cascade_with_chunks(db: Database) -> None:
    store = KnowledgeStore(db)
    (chunk_id,) = store.replace_source("a.md", [Chunk("a.md", "A", None, 0, "alpha")])
    store.store_embeddings("m", {chunk_id: [0.5, 0.25]})
    assert store.load_embeddings("m")[0][1].tolist() == [0.5, 0.25]
    assert store.load_embeddings("other") == []
    store.replace_source("a.md", [])
    assert store.load_embeddings("m") == []


def test_conversation_memory_window_and_alternation(db: Database) -> None:
    conv = ConversationStore(db)
    for i in range(5):
        conv.add_exchange("s", "web", f"q{i}", f"a{i}", Outcome.ANSWERED)
    recent = conv.recent("s", 2)
    assert [m.content for m in recent] == ["q3", "a3", "q4", "a4"]
    assert recent[0].role is Role.USER
    assert conv.recent("s", 0) == []
    assert conv.recent("unknown", 3) == []
    assert len(conv.transcript("s")) == 10


def test_consecutive_failures_counts_only_the_trailing_streak(db: Database) -> None:
    conv = ConversationStore(db)
    for outcome in [Outcome.UNANSWERED, Outcome.ANSWERED, Outcome.ERROR, Outcome.UNANSWERED]:
        conv.add_exchange("s", "web", "q", "a", outcome)
    assert conv.consecutive_failures("s", 5) == 2
    assert conv.consecutive_failures("s", 1) == 1
    assert conv.consecutive_failures("none", 3) == 0


def test_ticket_store_listing_and_active_lookup(db: Database) -> None:
    tickets = TicketStore(db)
    first = tickets.create("s1", "web", HandoffReason.LOW_CONFIDENCE, "q1")
    second = tickets.create("s2", "web", HandoffReason.USER_REQUEST, "q2")
    assert [t.id for t in tickets.list_tickets()] == [second.id, first.id]
    assert tickets.find_active("s1") is not None
    assert tickets.find_active("missing") is None
    assert tickets.get(404) is None
    from support_assistant.domain import TicketStatus

    tickets.set_status(first.id, TicketStatus.CLOSED)
    assert tickets.find_active("s1") is None
    assert [t.id for t in tickets.list_tickets(TicketStatus.OPEN)] == [second.id]


def test_blank_values_from_env_example_count_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ADMIN_TOKEN=\nANTHROPIC_API_KEY=\nSMTP_PORT=\nTELEGRAM_OPERATOR_CHAT_ID=\n"
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    s = Settings(_env_file=env_file)
    assert s.admin_token is None
    assert s.anthropic_api_key is None
    assert s.telegram_bot_token is None
    assert s.telegram_operator_chat_id is None
    assert s.smtp_port == 587


def test_env_example_parses_to_safe_defaults() -> None:
    example = Path(__file__).resolve().parent.parent / ".env.example"
    s = Settings(_env_file=example)
    assert s.admin_token is None
    assert s.llm_provider == "anthropic"
