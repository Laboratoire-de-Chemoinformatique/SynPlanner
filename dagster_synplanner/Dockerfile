# Multi-stage Dockerfile for SynPlanner Dagster platform
# Builds: dagster webserver, API server, and worker (user code) images

FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --locked --no-install-project

# Copy full project
COPY . .
RUN uv sync --locked --no-editable

# Install Dagster + orchestration extras
RUN pip install \
    dagster \
    dagster-webserver \
    dagster-postgres \
    dagster-docker \
    fastapi \
    uvicorn[standard] \
    httpx

# --- Dagster webserver image ---
FROM base AS dagster-webserver
EXPOSE 3000
ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p $DAGSTER_HOME
COPY dagster_synplanner/deploy/dagster.yaml $DAGSTER_HOME/dagster.yaml
CMD ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000"]

# --- User code (gRPC server for Dagster) ---
FROM base AS dagster-user-code
EXPOSE 4000
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "dagster_synplanner.repository"]

# --- API server ---
FROM base AS api-server
EXPOSE 8000
CMD ["uvicorn", "dagster_synplanner.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
