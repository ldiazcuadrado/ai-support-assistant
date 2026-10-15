from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from support_assistant.errors import RateLimitedError


class RateLimiter:
    """Sliding-window limiter held in process memory.

    Limits are per process: run a shared store behind the same interface if you scale out.
    """

    def __init__(
        self,
        max_events: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_events
        self._window = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._checks = 0

    def check(self, key: str) -> None:
        """Record one event for ``key`` or raise ``RateLimitedError``."""
        now = self._clock()
        events = self._events.setdefault(key, deque())
        cutoff = now - self._window
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self._max:
            raise RateLimitedError(retry_after=events[0] + self._window - now)
        events.append(now)
        self._checks += 1
        if self._checks % 1000 == 0:
            self._prune(cutoff)

    def _prune(self, cutoff: float) -> None:
        stale = [k for k, ev in self._events.items() if not ev or ev[-1] <= cutoff]
        for key in stale:
            del self._events[key]
