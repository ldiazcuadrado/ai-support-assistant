# syntax=docker/dockerfile:1

FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

ARG WITH_EMBEDDINGS=false
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $([ "$WITH_EMBEDDINGS" = "true" ] && echo --extra embeddings)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $([ "$WITH_EMBEDDINGS" = "true" ] && echo --extra embeddings)

FROM python:3.14-slim AS runtime
RUN useradd --system --uid 10001 --home-dir /app app \
    && mkdir /data \
    && chown app:app /data
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY knowledge_base ./knowledge_base

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/data/assistant.db \
    HOST=0.0.0.0 \
    PORT=8000

USER app
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=3)"]

ENTRYPOINT ["support-assistant"]
CMD ["serve"]
