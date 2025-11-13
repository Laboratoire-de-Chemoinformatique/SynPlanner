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

# Explicitly disable CUDA discovery inside the container
ENV CUDA_VISIBLE_DEVICES=-1

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
         build-essential python3-dev g++ curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

# Enable 'poetry export' via official plugin
RUN poetry self add poetry-plugin-export || "$POETRY_HOME/bin/python" -m pip install --no-cache-dir poetry-plugin-export

# 3. Copy lockfiles and export requirements
COPY pyproject.toml poetry.lock /app/
RUN poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt \
    && rm -rf /tmp/poetry-cache /tmp/pip-cache

# 4. Install Torch explicitly (CPU by default) and remaining deps via pip
ARG TORCH_VERSION=2.9.*
ARG TORCH_CHANNEL=cpu
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL} "torch==${TORCH_VERSION}"
RUN sed -i '/^torch[<>=!]/d' /tmp/requirements.txt || true \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# 5. Copy application code and install the SynPlanner package itself
COPY synplan /app/synplan
COPY README.rst /app/
RUN pip install --no-cache-dir .

# 6. Remove build tools and clean up
RUN apt-get purge -y --auto-remove build-essential python3-dev g++ curl \
    && rm -rf /var/lib/apt/lists/*

# 7. Entry point remains the SynPlanner CLI
ENTRYPOINT ["synplan"]