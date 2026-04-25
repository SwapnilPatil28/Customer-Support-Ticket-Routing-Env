"""Structured JSON logging for the environment server.

Every emitted log entry is one JSON object per line so it can be ingested by
standard log aggregators (Cloud Logging, Loki, Datadog, ELK) without extra
parsing.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Mapping

_LOGGER_CONFIGURED = False


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int((record.created % 1) * 1000):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, Mapping):
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", structured: bool = True) -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    if structured:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level.upper())
    _LOGGER_CONFIGURED = True


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra={"extra_fields": fields})
