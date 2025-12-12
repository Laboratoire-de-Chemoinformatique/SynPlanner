# syntax=docker/dockerfile:1.9
FROM python:3.12-slim AS builder

# Bring in uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure uv to build into /app using the system Python
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app

# Pre-sync dependencies with CPU-only torch (from optional-dependencies)
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync \
        --locked \
        --extra cpu \
        --no-install-project

# Install the project into the prepared environment
COPY . /src
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache \
    uv sync \
        --locked \
        --extra cpu \
        --no-editable

FROM python:3.12-slim

ENV PATH=/app/bin:$PATH

WORKDIR /app

COPY --from=builder /app /app

# Keep python shims aligned with the base image interpreter
RUN rm -f /app/bin/python /app/bin/python3 /app/bin/python3.12 2>/dev/null || true && \
    ln -s /usr/local/bin/python3 /app/bin/python && \
    ln -s /usr/local/bin/python3 /app/bin/python3 && \
    ln -s /usr/local/bin/python3 /app/bin/python3.12

ENTRYPOINT ["synplan"]