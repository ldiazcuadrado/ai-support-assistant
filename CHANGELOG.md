# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-15

### Added

- Ingestion CLI for Markdown, plain text, PDF and HTML files, single web pages and sitemaps, with
  overlapping chunks and source, title and page metadata stored in SQLite.
- Hybrid retrieval: BM25 through SQLite FTS5 by default, an optional embeddings backend behind an
  `Embedder` interface (fastembed extra), and reciprocal rank fusion when both are enabled.
- Grounded answering with Claude through the official Anthropic SDK, numbered source citations,
  an explicit "no answer" path, prompt caching and streamed responses.
- Human handoff on low retrieval confidence, an explicit request for a person, or repeated
  failures. Tickets are stored in SQLite and operators are notified through Telegram and/or email.
  Operators reply through the admin API or the CLI.
- Telegram bot channel and an HTTP API with an embeddable chat widget that streams over SSE.
- Per-session conversation memory, conversation logs and an analytics endpoint reporting
  questions per day, handoff rate and unanswered questions.
- Per-session and per-IP rate limiting, input length limits and prompt-injection hygiene.
- Sample knowledge base for a fictional bike repair shop.
- Dockerfile, Docker Compose file, GitHub Actions CI and Dependabot configuration.

[Unreleased]: https://github.com/ldiazcuadrado/ai-support-assistant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ldiazcuadrado/ai-support-assistant/releases/tag/v0.1.0
