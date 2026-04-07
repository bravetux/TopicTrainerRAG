# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 07 April 2026
# =============================================================================

# ── Stage: pull uv binary ─────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.6.14 AS uv

# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Copy uv binary from official image
COPY --from=uv /uv /bin/uv

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Environment ───────────────────────────────────────────────────────────────
# PYTHONUNBUFFERED: flush stdout/stderr immediately to CloudWatch Logs
# UV_COMPILE_BYTECODE: pre-compile .pyc files at build time to reduce startup latency
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy lockfile and project metadata first to leverage Docker layer caching.
# If pyproject.toml and uv.lock are unchanged, this layer is reused on rebuild.
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# ── Application source ────────────────────────────────────────────────────────
COPY src/ ./src/
COPY app.py ./

# ── Data directory ────────────────────────────────────────────────────────────
# Create the mount point for the EFS volume.
# In ECS, this is overridden by the efsVolumeConfiguration in the task definition.
# At runtime the EFS volume at /app/data provides:
#   /app/data/chroma/      — ChromaDB vector store
#   /app/data/sessions/    — chat session files
#   /app/data/generated/   — AI-generated training modules
#   /app/data/progress.db  — SQLite progress database
#   /app/data/documents/   — uploaded training documents
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# ── Runtime user ──────────────────────────────────────────────────────────────
USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
# Streamlit exposes a built-in health endpoint at /_stcore/health.
# ECS uses this to determine whether the task is healthy.
#   --interval=30s   check every 30 seconds
#   --timeout=10s    fail the check if no response within 10 seconds
#   --start-period=30s  give Streamlit time to start before counting failures
#   --retries=3      mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
      "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# --server.address 0.0.0.0   bind to all interfaces (required for ECS port mapping)
# --server.port 8501         explicit port
# --server.headless true     suppress browser-open prompt (no browser in container)
# --server.fileWatcherType none  disable inotify file watching (not needed in production)
ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", \
            "--server.address", "0.0.0.0", \
            "--server.port", "8501", \
            "--server.headless", "true", \
            "--server.fileWatcherType", "none"]
