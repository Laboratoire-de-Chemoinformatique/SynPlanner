# syntax=docker/dockerfile:1.9
FROM python:3.13-slim AS builder

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Override with: docker build --build-arg VERSION=1.2.3 .
ARG VERSION=0.0.0
ENV HATCH_BUILD_VERSION=${VERSION} \
    SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync \
        --locked \
        --no-install-project

COPY . /src
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache \
    uv sync \
        --locked \
        --no-editable

FROM python:3.13-slim

ENV PATH=/app/bin:$PATH

WORKDIR /app

COPY --from=builder /app /app

RUN rm -f /app/bin/python /app/bin/python3 /app/bin/python3.13 2>/dev/null || true && \
    ln -s /usr/local/bin/python3 /app/bin/python && \
    ln -s /usr/local/bin/python3 /app/bin/python3 && \
    ln -s /usr/local/bin/python3 /app/bin/python3.13

ENTRYPOINT ["fdc"]