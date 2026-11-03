# ── 1. Build the React frontend ──
# Debian-based (glibc) image: the native TypeScript 7 compiler and Vite's binaries ship glibc builds.
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# ── 2. Python backend (serves the API and the built frontend) ──
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS app
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

COPY src/ src/
COPY --from=frontend /frontend/dist frontend/dist
RUN uv sync --no-dev --frozen

RUN useradd --create-home --uid 1000 app && mkdir -p /data/papers && chown -R app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
# Run the venv's uvicorn directly (no uv at runtime: nothing needs writing outside /data).
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
