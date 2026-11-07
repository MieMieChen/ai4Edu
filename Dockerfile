FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.5 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN useradd --uid 10001 --create-home grader && mkdir -p /app/data/assets && chown -R grader:grader /app/data
USER grader
EXPOSE 8000
CMD ["uvicorn", "ai4edu.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
