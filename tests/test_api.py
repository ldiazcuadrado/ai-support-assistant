from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from support_assistant.channels.api import create_app
from support_assistant.factory import Container
from support_assistant.llm import FakeLLM

SESSION = "abcdef123456"
ADMIN = {"Authorization": "Bearer secret-token"}
QUESTION = "What does a puncture repair cost?"


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as c:
        yield c


def _chat(client: TestClient, message: str, session: str = SESSION) -> dict[str, object]:
    response = client.post("/api/chat", json={"session_id": session, "message": message})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def _sse_events(text: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in filter(None, text.split("\n\n")):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_chat_returns_grounded_answer_with_citations(client: TestClient) -> None:
    body = _chat(client, QUESTION)
    assert body["outcome"] == "answered"
    assert body["citations"] and body["citations"][0]["index"] == 1  # type: ignore[index]
    assert body["ticket_id"] is None


def test_chat_handoff_creates_ticket_visible_to_admin(client: TestClient) -> None:
    body = _chat(client, "I need to speak to a human")
    assert body["outcome"] == "handoff"
    tickets = client.get("/admin/tickets", headers=ADMIN).json()
    assert [t["id"] for t in tickets] == [body["ticket_id"]]
    detail = client.get(f"/admin/tickets/{body['ticket_id']}", headers=ADMIN).json()
    assert detail["messages"][0]["author"] == "customer"
    only_open = client.get("/admin/tickets?status=closed", headers=ADMIN).json()
    assert only_open == []


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "short", "message": "hi"},
        {"session_id": SESSION, "message": ""},
        {"session_id": "bad session id!!", "message": "hi"},
        {"message": "hi"},
    ],
)
def test_invalid_payloads_are_422(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/api/chat", json=payload).status_code == 422


def test_message_over_configured_limit_is_rejected(client: TestClient) -> None:
    response = client.post("/api/chat", json={"session_id": SESSION, "message": "x" * 1500})
    assert response.status_code == 422
    assert "longer" in response.json()["detail"]


def test_session_rate_limit_returns_429_with_retry_after(
    make_container: Callable[..., Container],
) -> None:
    c = make_container(rate_limit_requests=2)
    with TestClient(create_app(c)) as client:
        for _ in range(2):
            assert (
                client.post(
                    "/api/chat", json={"session_id": SESSION, "message": "human"}
                ).status_code
                == 200
            )
        limited = client.post("/api/chat", json={"session_id": SESSION, "message": "human"})
        assert limited.status_code == 429
        assert int(limited.headers["Retry-After"]) >= 1


def test_ip_rate_limit_applies_across_sessions(
    make_container: Callable[..., Container],
) -> None:
    c = make_container(rate_limit_requests=1)  # ip limit is 3x
    with TestClient(create_app(c)) as client:
        codes = [
            client.post(
                "/api/chat", json={"session_id": f"session-{i:04d}", "message": "human"}
            ).status_code
            for i in range(4)
        ]
    assert codes == [200, 200, 200, 429]


def test_stream_emits_tokens_then_done(client: TestClient) -> None:
    response = client.post("/api/chat/stream", json={"session_id": SESSION, "message": QUESTION})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    names = [n for n, _ in events]
    assert names[-1] == "done" and set(names[:-1]) == {"token"}
    streamed = "".join(str(d["text"]) for n, d in events if n == "token")
    assert streamed == events[-1][1]["answer"]
    assert events[-1][1]["outcome"] == "answered"


def test_stream_validation_errors_are_http_errors(client: TestClient) -> None:
    response = client.post("/api/chat/stream", json={"session_id": SESSION, "message": "x" * 1500})
    assert response.status_code == 422


def test_stream_unexpected_failure_becomes_error_event(
    make_container: Callable[..., Container],
) -> None:
    class Exploding(FakeLLM):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            yield "partial "
            raise RuntimeError("socket closed")

    from support_assistant.ingestion import IngestPipeline
    from tests.conftest import KNOWLEDGE_BASE

    c = make_container(Exploding())
    IngestPipeline(c.knowledge, 900, 150).ingest_path(KNOWLEDGE_BASE)
    with TestClient(create_app(c)) as client:
        response = client.post(
            "/api/chat/stream", json={"session_id": SESSION, "message": QUESTION}
        )
    names = [n for n, _ in _sse_events(response.text)]
    assert names == ["token", "error"]


def test_operator_reply_round_trip(client: TestClient) -> None:
    ticket_id = _chat(client, "human please")["ticket_id"]
    reply = client.post(
        f"/admin/tickets/{ticket_id}/reply", headers=ADMIN, json={"message": "We open at 8."}
    )
    assert reply.status_code == 200
    assert reply.json()["delivered"] is False

    replies = client.get(f"/api/sessions/{SESSION}/operator-replies").json()
    assert [r["text"] for r in replies] == ["We open at 8."]
    assert (
        client.get(f"/api/sessions/{SESSION}/operator-replies?after={replies[0]['id']}").json()
        == []
    )
    assert client.get("/api/sessions/otheruser123/operator-replies").json() == []

    closed = client.post(f"/admin/tickets/{ticket_id}/close", headers=ADMIN)
    assert closed.json() == {"status": "closed"}


def test_unknown_ticket_is_404(client: TestClient) -> None:
    assert client.get("/admin/tickets/999", headers=ADMIN).status_code == 404
    assert (
        client.post("/admin/tickets/999/reply", headers=ADMIN, json={"message": "x"}).status_code
        == 404
    )
    assert client.post("/admin/tickets/999/close", headers=ADMIN).status_code == 404


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "secret-token-x"}]
)
def test_admin_requires_valid_token(client: TestClient, headers: dict[str, str]) -> None:
    for path in ("/admin/tickets", "/admin/analytics", "/admin/sessions/x/messages"):
        assert client.get(path, headers=headers).status_code == 401


def test_admin_is_disabled_without_a_token(make_container: Callable[..., Container]) -> None:
    c = make_container(admin_token=None)
    with TestClient(create_app(c)) as client:
        assert client.get("/admin/tickets", headers=ADMIN).status_code == 503


def test_blank_admin_token_never_authenticates(
    make_container: Callable[..., Container],
) -> None:
    c = make_container(admin_token=SecretStr(""))
    with TestClient(create_app(c)) as client:
        response = client.get("/admin/tickets", headers={"Authorization": "Bearer "})
    assert response.status_code == 503


def test_analytics_and_transcript(client: TestClient) -> None:
    _chat(client, QUESTION)
    _chat(client, "Do you sell pizza and sushi?", session="session-two-0001")
    _chat(client, "talk to a human", session="session-three-001")
    report = client.get("/admin/analytics?days=7", headers=ADMIN).json()
    assert report["total_questions"] == 3
    assert report["answered"] == 1
    assert report["handoffs"] == 2
    assert report["handoff_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert [q["question"] for q in report["unanswered_questions"]] == [
        "Do you sell pizza and sushi?"
    ]
    assert sum(report["questions_per_day"].values()) == 3

    log = client.get(f"/admin/sessions/web:{SESSION}/messages", headers=ADMIN).json()
    assert [m["role"] for m in log] == ["user", "assistant"]


def test_widget_assets_and_demo_page(client: TestClient) -> None:
    js = client.get("/widget.js")
    assert js.status_code == 200 and "text/javascript" in js.headers["content-type"]
    assert "/api/chat/stream" in js.text
    css = client.get("/widget.css")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    page = client.get("/")
    assert page.status_code == 200 and '<script src="/widget.js"' in page.text


def test_cors_is_opt_in(make_container: Callable[..., Container]) -> None:
    origin = {"Origin": "https://shop.example"}
    with TestClient(create_app(make_container())) as client:
        assert "access-control-allow-origin" not in client.get("/health", headers=origin).headers
    c = make_container(cors_origins="https://shop.example, https://other.example")
    with TestClient(create_app(c)) as client:
        allowed = client.get("/health", headers=origin)
        assert allowed.headers["access-control-allow-origin"] == "https://shop.example"
        assert (
            "access-control-allow-origin"
            not in client.get("/health", headers={"Origin": "https://evil.example"}).headers
        )
