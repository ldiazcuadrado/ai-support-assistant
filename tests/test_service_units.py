from __future__ import annotations

import pytest

from support_assistant.domain import HandoffReason, TicketStatus
from support_assistant.errors import InputRejectedError, RateLimitedError, TicketNotFoundError
from support_assistant.factory import Container
from support_assistant.notify import CompositeNotifier
from support_assistant.service import HandoffService, RateLimiter, sanitize_input, wants_human
from tests.conftest import RecordingDelivery, RecordingNotifier


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_blocks_then_recovers() -> None:
    clock = Clock()
    limiter = RateLimiter(2, 10, clock)
    limiter.check("a")
    clock.now = 1
    limiter.check("a")
    with pytest.raises(RateLimitedError) as info:
        limiter.check("a")
    assert info.value.retry_after == pytest.approx(9)
    limiter.check("b")
    clock.now = 10.5
    limiter.check("a")


def test_rate_limiter_prunes_idle_keys() -> None:
    clock = Clock()
    limiter = RateLimiter(1, 1, clock)
    for i in range(999):
        limiter.check(f"k{i}")
    clock.now = 100
    limiter.check("last")
    assert list(limiter._events) == ["last"]


@pytest.mark.parametrize(
    "text",
    [
        "I want to talk to a human",
        "Can I speak to someone?",
        "get me an AGENT",
        "let me talk with a person",
        "is there a live chat",
        "connect me to the manager",
    ],
)
def test_wants_human_positive(text: str) -> None:
    assert wants_human(text)


@pytest.mark.parametrize(
    "text",
    ["How much is a service?", "what are your hours", "my chain keeps slipping"],
)
def test_wants_human_negative(text: str) -> None:
    assert not wants_human(text)


def test_sanitize_input() -> None:
    assert sanitize_input("  hi\x00 there\r\n\n\n\n\nbye ", 100) == "hi there\n\nbye"
    with pytest.raises(InputRejectedError):
        sanitize_input("a" * 11, 10)
    with pytest.raises(InputRejectedError):
        sanitize_input("\x01\x02", 10)


async def test_handoff_lifecycle(container: Container) -> None:
    notifier, delivery = RecordingNotifier(), RecordingDelivery()
    service = HandoffService(container.tickets, notifier, delivery)

    ticket, is_new = await service.open("web:abc", "web", HandoffReason.USER_REQUEST, "help")
    assert is_new
    again, is_new = await service.open("web:abc", "web", HandoffReason.USER_REQUEST, "more")
    assert not is_new and again.id == ticket.id
    assert len(notifier.tickets) == 1

    message, delivered = await service.reply(ticket.id, "On it")
    assert delivered and message.author == "operator"
    stored = container.tickets.get(ticket.id)
    assert stored is not None and stored.status is TicketStatus.ANSWERED

    reopened, is_new = await service.open(
        "web:abc", "web", HandoffReason.USER_REQUEST, "still broken"
    )
    assert not is_new and reopened.id == ticket.id
    assert len(notifier.tickets) == 2
    refreshed = container.tickets.get(ticket.id)
    assert refreshed is not None and refreshed.status is TicketStatus.OPEN

    await service.close(ticket.id)
    fresh, is_new = await service.open("web:abc", "web", HandoffReason.USER_REQUEST, "new issue")
    assert is_new and fresh.id != ticket.id


async def test_handoff_survives_notifier_and_delivery_failures(container: Container) -> None:
    class Broken:
        async def notify_handoff(self, ticket: object) -> None:
            raise RuntimeError("smtp down")

        async def deliver(self, ticket: object, text: str) -> bool:
            raise RuntimeError("telegram down")

    service = HandoffService(container.tickets, Broken(), Broken())
    ticket, _ = await service.open("s", "web", HandoffReason.LOW_CONFIDENCE, "q")
    _, delivered = await service.reply(ticket.id, "hello")
    assert not delivered


async def test_handoff_without_notifier_still_creates_ticket(container: Container) -> None:
    service = HandoffService(container.tickets)
    ticket, is_new = await service.open("s", "web", HandoffReason.LOW_CONFIDENCE, "q")
    assert is_new
    _, delivered = await service.reply(ticket.id, "hi")
    assert not delivered


async def test_unknown_ticket(container: Container) -> None:
    with pytest.raises(TicketNotFoundError):
        await container.handoff.reply(999, "x")
    with pytest.raises(TicketNotFoundError):
        await container.handoff.close(999)


async def test_composite_notifier_isolates_failures(container: Container) -> None:
    class Failing:
        async def notify_handoff(self, ticket: object) -> None:
            raise RuntimeError("nope")

    good = RecordingNotifier()
    ticket = container.tickets.create("s", "web", HandoffReason.LOW_CONFIDENCE, "q")
    await CompositeNotifier([Failing(), good]).notify_handoff(ticket)
    assert len(good.tickets) == 1
