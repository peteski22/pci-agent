FROM python:3.13-alpine AS builder

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for Python packages with C extensions
RUN apk add --no-cache \
    gcc \
    musl-dev \
    postgresql-dev \
    libffi-dev \
    openssl-dev \
    curl

# Install uv
# Ref: https://docs.astral.sh/uv/getting-started/installation/
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1

# Copy project files first
COPY pyproject.toml README.md ./
COPY uv.lock* ./

# Install dependencies (generate lock if missing)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Copy source
COPY src ./src

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Production image
FROM python:3.13-alpine

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apk add --no-cache \
    libpq \
    libffi \
    openssl \
    curl

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src

# Models directory (mount at runtime)
VOLUME /app/models

# Default port for agent API
EXPOSE 8082

CMD ["python", "-m", "pci_agent"]
