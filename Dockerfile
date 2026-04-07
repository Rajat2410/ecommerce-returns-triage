FROM python:3.10-slim


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


COPY pyproject.toml uv.lock ./


RUN uv sync --frozen --no-cache


COPY . .


EXPOSE 7860
ENV PYTHONPATH=/app
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]