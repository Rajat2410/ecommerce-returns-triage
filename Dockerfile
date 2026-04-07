# Use the same slim base image
FROM python:3.10-slim

# MANDATORY: Install uv inside the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1. Copy only the lockfile and pyproject first 
# This allows Docker to cache the 'install' layer (Fast builds!)
COPY pyproject.toml uv.lock ./

# 2. Install dependencies strictly from the lockfile
# --frozen prevents pip-style "backtracking" or version guessing
RUN uv sync --frozen --no-cache

# 3. Copy the rest of the application code
COPY . .

# 4. Environment Configuration
EXPOSE 7860
ENV PYTHONPATH=/app
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"

# 5. Start the FastAPI server using 'uv run'
# This ensures uvicorn is executed within the synced environment
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]