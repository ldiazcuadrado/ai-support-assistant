class SupportAssistantError(Exception):
    """Base class for errors raised by this package."""


class ConfigError(SupportAssistantError):
    """The configuration is missing or inconsistent."""


class LLMError(SupportAssistantError):
    """The language model could not produce a response."""


class IngestError(SupportAssistantError):
    """A document could not be loaded or parsed."""


class InputRejectedError(SupportAssistantError):
    """User input was refused before reaching the model."""


class RateLimitedError(SupportAssistantError):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limit exceeded, retry in {retry_after:.0f}s")
        self.retry_after = retry_after


class TicketNotFoundError(SupportAssistantError):
    """No ticket exists with the requested id."""
