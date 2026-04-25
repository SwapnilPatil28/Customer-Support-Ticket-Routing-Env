# Root Dockerfile kept for compatibility with tools that expect it at
# the repository root. Mirrors server/Dockerfile but uses the top-level
# requirements.txt so integrators can run a fuller image if desired.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENABLE_WEB_INTERFACE=true \
    ENV_LOG_LEVEL=INFO \
    ENV_STRUCTURED_LOGGING=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/server/requirements.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
