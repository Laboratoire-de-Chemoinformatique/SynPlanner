# Use a slim Python base image
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
    PIP_CACHE_DIR="/tmp/pip-cache"

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# 2. Install build tools & Poetry
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
         build-essential python3-dev g++ curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

# 3. Copy only dependency-defining files to leverage Docker cache
COPY pyproject.toml poetry.lock /app/

# 4. Install project dependencies (no-root means not installing the project itself yet)
RUN poetry install --without dev --no-root --no-interaction && rm -rf /tmp/poetry-cache /tmp/pip-cache

# 5. Copy the application code into the container
# This copies your main application logic
COPY synplan /app/synplan
COPY README.rst /app/

# The second install will now install the synplan package itself
RUN poetry install --without dev --no-interaction && rm -rf /tmp/poetry-cache /tmp/pip-cache

# 6. Remove build tools to keep the final image slim
RUN apt-get purge -y --auto-remove build-essential python3-dev g++ curl \
    && rm -rf /var/lib/apt/lists/*

# 7. Expose the default Streamlit port (8501)
EXPOSE 8501

# 8. Add a health check to ensure the Streamlit app is responsive
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 9. Set the entry point to run the Streamlit GUI script
# This is the key change to run your GUI instead of the CLI
ENTRYPOINT ["streamlit", "run", "synplan/interfaces/gui.py", "--server.port=8501", "--server.address=0.0.0.0"]