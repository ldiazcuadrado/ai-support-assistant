from __future__ import annotations

import smtplib
from collections.abc import AsyncIterator
from email.message import EmailMessage
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import anthropic
import httpx
import pytest
from pydantic import SecretStr

from support_assistant.config import Settings
from support_assistant.domain import HandoffReason, Message, Role, Ticket
from support_assistant.errors import ConfigError, LLMError
from support_assistant.factory import Container, create_llm, create_notifier
from support_assistant.llm import AnthropicLLM, FakeLLM, LLMRequest
from support_assistant.notify import CompositeNotifier, EmailNotifier, SmtpConfig


class FakeSMTP:
    instances: ClassVar[list[FakeSMTP]] = []

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host, self.port = host, port
        self.calls: list[str] = []
        self.sent: list[EmailMessage] = []
        FakeSMTP.instances.append(self)

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def starttls(self) -> None:
        self.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.calls.append(f"login:{user}:{password}")

    def send_message(self, message: EmailMessage) -> None:
        self.sent.append(message)


@pytest.fixture
def ticket(container: Container) -> Ticket:
    return container.tickets.create("web:abc", "web", HandoffReason.LOW_CONFIDENCE, "Where is it?")


async def test_email_notifier(monkeypatch: pytest.MonkeyPatch, ticket: Ticket) -> None:
    FakeSMTP.instances.clear()
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    config = SmtpConfig("smtp.test", 2525, "bot@test", "ops@test", "user", "pw")
    await EmailNotifier(config).notify_handoff(ticket)
    (smtp,) = FakeSMTP.instances
    assert (smtp.host, smtp.port) == ("smtp.test", 2525)
    assert smtp.calls == ["starttls", "login:user:pw"]
    message = smtp.sent[0]
    assert message["To"] == "ops@test" and message["From"] == "bot@test"
    assert f"#{ticket.id}" in message["Subject"]
    assert "Where is it?" in message.get_content()


async def test_email_without_auth_or_tls(monkeypatch: pytest.MonkeyPatch, ticket: Ticket) -> None:
    FakeSMTP.instances.clear()
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    config = SmtpConfig("localhost", 25, "a@test", "b@test", starttls=False)
    await EmailNotifier(config).notify_handoff(ticket)
    assert FakeSMTP.instances[0].calls == []


def test_create_notifier_from_settings(settings: Settings) -> None:
    async def sender(chat_id: int | str, text: str) -> None:
        return None

    assert create_notifier(settings, sender) is None
    both = settings.model_copy(
        update={
            "telegram_operator_chat_id": "1",
            "smtp_host": "h",
            "operator_email": "ops@test",
            "smtp_password": SecretStr("pw"),
        }
    )
    assert isinstance(create_notifier(both, sender), CompositeNotifier)
    assert create_notifier(both, None) is not None
    assert (
        create_notifier(settings.model_copy(update={"telegram_operator_chat_id": "1"}), None)
        is None
    )


def test_create_llm(settings: Settings) -> None:
    assert isinstance(create_llm(settings), FakeLLM)
    real = settings.model_copy(update={"llm_provider": "anthropic"})
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_llm(real)
    configured = real.model_copy(update={"anthropic_api_key": SecretStr("sk-test")})
    assert isinstance(create_llm(configured), AnthropicLLM)


class _Stream:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            for piece in self._pieces:
                yield piece

        return gen()


class _Messages:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        content = [
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="tool", text="x"),
        ]
        return SimpleNamespace(content=content, usage=usage)

    def stream(self, **kwargs: Any) -> _Stream:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _Stream(["a", "b"])


def _llm(messages: _Messages) -> AnthropicLLM:
    return AnthropicLLM("key", "claude-sonnet-5-5", client=SimpleNamespace(messages=messages))  # type: ignore[arg-type]


def _request(history: bool) -> LLMRequest:
    messages = [Message(Role.USER, "final")]
    if history:
        messages = [Message(Role.USER, "q1"), Message(Role.ASSISTANT, "a1"), *messages]
    return LLMRequest("SYSTEM", tuple(messages), 256)


async def test_anthropic_request_shape_and_prompt_caching() -> None:
    messages = _Messages()
    text = await _llm(messages).complete(_request(history=True))
    assert text == "Hello "
    call = messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["max_tokens"] == 256
    assert call["system"] == [
        {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"][0] == {"role": "user", "content": "q1"}
    assert call["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"][2] == {"role": "user", "content": "final"}


async def test_anthropic_single_message_has_no_history_breakpoint() -> None:
    messages = _Messages()
    await _llm(messages).complete(_request(history=False))
    assert messages.calls[0]["messages"] == [{"role": "user", "content": "final"}]


async def test_anthropic_streaming() -> None:
    messages = _Messages()
    assert [p async for p in _llm(messages).stream(_request(False))] == ["a", "b"]


async def test_anthropic_errors_are_wrapped() -> None:
    error = anthropic.APIConnectionError(
        request=cast("Any", httpx.Request("POST", "https://api.test"))
    )
    with pytest.raises(LLMError):
        await _llm(_Messages(error)).complete(_request(False))
    with pytest.raises(LLMError):
        [p async for p in _llm(_Messages(error)).stream(_request(False))]


async def test_fake_llm_behaviour() -> None:
    llm = FakeLLM(responses=["one two", "last"])
    request = _request(False)
    assert await llm.complete(request) == "one two"
    assert [p async for p in llm.stream(request)] == ["last"]
    assert await llm.complete(request) == "last"
    assert len(llm.requests) == 3
    assert await FakeLLM().complete(request) == "NO_ANSWER"
    with pytest.raises(LLMError):
        await FakeLLM(error=ValueError("x")).complete(request)


async def test_fake_llm_extracts_the_line_that_matches_the_question(container: Container) -> None:
    from support_assistant.domain import ScoredChunk
    from support_assistant.llm import render_user_message

    chunk = container.retriever.search("puncture repair cost", 1)[0].chunk
    sources = (ScoredChunk(chunk, 1.0, 1.0),)
    prompt = render_user_message("How much is a gear cable replacement?", sources)
    request = LLMRequest("s", (Message(Role.USER, prompt),), 100, sources)
    answer = await FakeLLM().complete(request)
    assert answer == "Gear cable replacement: 22 per cable [1]"
