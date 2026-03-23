"""Application logging configuration.

Design choice:
- Configure logging once at process startup so module loggers stay lightweight and consistent.

Tradeoff:
- A single global configuration is simple, but less flexible than per-request structured logging.

Production caveat:
- Real services usually add correlation IDs, JSON formatting, and sink-specific handlers.
"""

from __future__ import annotations

import json as _json
import logging
import os


QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "qdrant_client",
)

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return _json.dumps(entry, default=str)


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "text").strip().lower()

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when called more than once (e.g. tests)
    if not root.handlers:
        handler = logging.StreamHandler()
        if log_format == "json":
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(TEXT_FORMAT))
        root.addHandler(handler)
