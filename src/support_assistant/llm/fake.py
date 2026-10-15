from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence

from support_assistant.errors import LLMError
from support_assistant.llm.base import LLMRequest
from support_assistant.llm.prompts import NO_ANSWER
from support_assistant.retrieval.text import stems

_UNIT = re.compile(r"(?<=[.!?])\s+|\n+")
_QUESTION = re.compile(r"<customer_question>\n(.*?)\n</customer_question>", re.DOTALL)
_WORD = re.compile(r"\S+\s*")


class FakeLLM:
    """Deterministic offline stand-in implementing ``LLMClient``.

    Without scripted ``responses`` it answers extractively from the first retrieved source, or
    replies ``NO_ANSWER`` when there is none. Every request is recorded in ``requests``.
    """

    def __init__(self, responses: Sequence[str] = (), error: Exception | None = None) -> None:
        self.requests: list[LLMRequest] = []
        self._responses = list(responses)
        self._error = error

    async def complete(self, request: LLMRequest) -> str:
        self.requests.append(request)
        if self._error is not None:
            raise self._error if isinstance(self._error, LLMError) else LLMError(str(self._error))
        if self._responses:
            return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return self._extract(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        for piece in _WORD.findall(await self.complete(request)):
            yield piece

    @staticmethod
    def _extract(request: LLMRequest) -> str:
        if not request.sources:
            return NO_ANSWER
        asked = _QUESTION.search(request.messages[-1].content)
        wanted = stems(asked.group(1)) if asked else set()
        text = request.sources[0].chunk.text.replace("|", ",")
        units = [u for u in (_clean(u) for u in _UNIT.split(text)) if len(u) > 15]
        if not units:
            return f"{_clean(text)} [1]"
        best = max(units, key=lambda unit: len(wanted & stems(unit)))
        return f"{best} [1]"


def _clean(text: str) -> str:
    return " ".join(text.strip(" ,-#*").split())
