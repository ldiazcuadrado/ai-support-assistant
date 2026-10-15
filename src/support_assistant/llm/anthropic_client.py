from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import anthropic
from anthropic.types import MessageParam, TextBlockParam

from support_assistant.errors import LLMError
from support_assistant.llm.base import LLMRequest

logger = logging.getLogger(__name__)


class AnthropicLLM:
    """Claude through the official SDK.

    The system prompt is stable across requests and carries a cache breakpoint. A second
    breakpoint on the last prior turn lets later turns of a conversation reuse the history.
    """

    def __init__(
        self, api_key: str, model: str, client: anthropic.AsyncAnthropic | None = None
    ) -> None:
        self._model = model
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, request: LLMRequest) -> str:
        system, messages = self._build(request)
        try:
            response = await self._client.messages.create(
                model=self._model, max_tokens=request.max_tokens, system=system, messages=messages
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        usage = response.usage
        logger.info(
            "llm_usage",
            extra={
                "model": self._model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_input_tokens,
                "cache_write_tokens": usage.cache_creation_input_tokens,
            },
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        system, messages = self._build(request)
        try:
            async with self._client.messages.stream(
                model=self._model, max_tokens=request.max_tokens, system=system, messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic stream failed: {exc}") from exc

    @staticmethod
    def _build(request: LLMRequest) -> tuple[list[TextBlockParam], list[MessageParam]]:
        system: list[TextBlockParam] = [
            {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
        ]
        messages: list[MessageParam] = []
        last_history = len(request.messages) - 2
        for i, message in enumerate(request.messages):
            if i == last_history:
                block: TextBlockParam = {
                    "type": "text",
                    "text": message.content,
                    "cache_control": {"type": "ephemeral"},
                }
                messages.append({"role": message.role.value, "content": [block]})
            else:
                messages.append({"role": message.role.value, "content": message.content})
        return system, messages
