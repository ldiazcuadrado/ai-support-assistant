from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from support_assistant.config import Settings
from support_assistant.errors import ConfigError
from support_assistant.llm import AnthropicLLM, FakeLLM, LLMClient
from support_assistant.notify import (
    CompositeNotifier,
    EmailNotifier,
    Notifier,
    ReplyDelivery,
    SmtpConfig,
    TelegramDelivery,
    TelegramNotifier,
    TelegramSender,
    bot_sender,
)
from support_assistant.retrieval import (
    Embedder,
    FastEmbedEmbedder,
    HybridRetriever,
    LexicalRetriever,
    Retriever,
    VectorRetriever,
)
from support_assistant.service import Assistant, AssistantConfig, HandoffService, RateLimiter
from support_assistant.storage import (
    AnalyticsStore,
    ConversationStore,
    Database,
    KnowledgeStore,
    TicketStore,
)


@dataclass
class Container:
    """The wired object graph. The assistant, and with it the LLM client, is built on first use,
    so commands that never answer questions do not need an API key."""

    settings: Settings
    database: Database
    knowledge: KnowledgeStore
    conversations: ConversationStore
    tickets: TicketStore
    analytics: AnalyticsStore
    handoff: HandoffService
    retriever: Retriever
    ip_limiter: RateLimiter
    embedder: Embedder | None
    llm: LLMClient | None = None

    @cached_property
    def assistant(self) -> Assistant:
        s = self.settings
        return Assistant(
            retriever=self.retriever,
            llm=self.llm or create_llm(s),
            conversations=self.conversations,
            handoff=self.handoff,
            limiter=RateLimiter(s.rate_limit_requests, s.rate_limit_window_seconds),
            config=AssistantConfig(
                business_name=s.business_name,
                top_k=s.retrieval_top_k,
                min_relevance=s.min_relevance,
                memory_turns=s.memory_turns,
                max_input_chars=s.max_input_chars,
                max_output_tokens=s.max_output_tokens,
                handoff_after_failures=s.handoff_after_failures,
            ),
        )


def create_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if settings.anthropic_api_key is None:
        raise ConfigError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
    return AnthropicLLM(settings.anthropic_api_key.get_secret_value(), settings.anthropic_model)


def create_embedder(settings: Settings) -> Embedder | None:
    if settings.embeddings_backend == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model)
    return None


def create_notifier(settings: Settings, sender: TelegramSender | None) -> Notifier | None:
    notifiers: list[Notifier] = []
    if sender is not None and settings.telegram_operator_chat_id:
        notifiers.append(TelegramNotifier(sender, settings.telegram_operator_chat_id))
    if settings.smtp_host and settings.operator_email:
        notifiers.append(
            EmailNotifier(
                SmtpConfig(
                    host=settings.smtp_host,
                    port=settings.smtp_port,
                    sender=settings.smtp_from or settings.operator_email,
                    recipient=settings.operator_email,
                    username=settings.smtp_username,
                    password=settings.smtp_password.get_secret_value()
                    if settings.smtp_password
                    else None,
                    starttls=settings.smtp_starttls,
                )
            )
        )
    return CompositeNotifier(notifiers) if notifiers else None


def build_container(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    sender: TelegramSender | None = None,
    notifier: Notifier | None = None,
    delivery: ReplyDelivery | None = None,
) -> Container:
    """Wire every component. Arguments override the defaults derived from ``settings``."""
    database = Database(settings.database_path)
    database.initialize()
    knowledge = KnowledgeStore(database)
    conversations = ConversationStore(database)
    tickets = TicketStore(database)

    embedder = embedder or create_embedder(settings)
    retrievers: list[Retriever] = [LexicalRetriever(knowledge)]
    if embedder is not None:
        retrievers.append(VectorRetriever(knowledge, embedder))

    if sender is None and settings.telegram_bot_token is not None:
        sender = bot_sender(settings.telegram_bot_token.get_secret_value())
    if delivery is None and sender is not None:
        delivery = TelegramDelivery(sender)
    handoff = HandoffService(tickets, notifier or create_notifier(settings, sender), delivery)

    return Container(
        settings=settings,
        database=database,
        knowledge=knowledge,
        conversations=conversations,
        tickets=tickets,
        analytics=AnalyticsStore(database),
        handoff=handoff,
        retriever=HybridRetriever(retrievers),
        ip_limiter=RateLimiter(
            settings.rate_limit_requests * 3, settings.rate_limit_window_seconds
        ),
        embedder=embedder,
        llm=llm,
    )
