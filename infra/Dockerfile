# ── Stage 1: Build Next.js static export ─────────────────────────────
FROM node:20-slim AS ui-builder

WORKDIR /ui

# Install deps first (layer cache — only re-runs when lock file changes)
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --prefer-offline

# Copy source and build
COPY ui/ .

# NEXT_PUBLIC_API_URL="" means relative URLs — UI and API share the same origin
# on Fly.io, so /analyze and /health resolve to the same host automatically.
ENV NEXT_PUBLIC_API_URL=""
# API key is baked into the static bundle at build time.
# Pass it via: fly deploy --build-arg NEXT_PUBLIC_API_KEY=<your-key>
# or set [build.args] in fly.toml.
ARG NEXT_PUBLIC_API_KEY=""
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY
RUN npm run build


# ── Stage 2: Python API ───────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY ariadne/ ariadne/
COPY pyproject.toml .
COPY data/ data/

# Install the project itself (editable not needed in prod)
RUN pip install --no-cache-dir -e .

# Copy built UI static files from Stage 1 into the Python image.
# FastAPI's StaticFiles mounts this at runtime — no Node.js needed at runtime.
COPY --from=ui-builder /ui/out ./ui/out

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]

CMD ["uvicorn", "ariadne.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
