# syntax=docker/dockerfile:1
FROM python:3.12-slim

# 1. Set environment variables for pip & Poetry
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR="/tmp/poetry-cache" \
    PIP_CACHE_DIR="/tmp/pip-cache" \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" \
    PIP_PREFER_BINARY=1

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# 2. Install build tools & Poetry, export cached dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
         build-essential python3-dev g++ curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

# 3. Copy lockfiles and install only dependencies (no root package)
COPY pyproject.toml poetry.lock /app/
RUN poetry install --without dev --no-root --no-interaction && rm -rf /tmp/poetry-cache /tmp/pip-cache

# 4. Copy application code and install the SynPlanner package itself
COPY synplan /app/synplan
COPY README.rst /app/
RUN poetry install --without dev --no-interaction && rm -rf /tmp/poetry-cache /tmp/pip-cache

# 5. Remove build tools and clean up
RUN apt-get purge -y --auto-remove build-essential python3-dev g++ curl \
    && rm -rf /var/lib/apt/lists/*

# 6. Entry point remains the SynPlanner CLI
ENTRYPOINT ["synplan"]