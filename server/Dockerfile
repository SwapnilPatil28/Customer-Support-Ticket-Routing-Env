# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Incident Command Center - OpenEnv server image
# -----------------------------------------------------------------------------
# Keeps the runtime image small (~150 MB) by installing only the server-side
# dependencies. Training dependencies ship via the top-level requirements.txt
# for Colab / local training.
# -----------------------------------------------------------------------------

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
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

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
