"""Logging helpers producing concise key=value output."""

from __future__ import annotations

import contextvars
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Mapping

REQUEST_ID_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="-",
)


class RequestIdFilter(logging.Filter):
    """Attach the request identifier to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_VAR.get("-")
        return True


class KeyValueFormatter(logging.Formatter):
    """Format log records as timestamped key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")
        kv_fields: dict[str, Any] = {}
        extra_fields = getattr(record, "kv", None)
        if isinstance(extra_fields, Mapping):
            kv_fields.update(extra_fields)
        if "event" not in kv_fields:
            kv_fields["event"] = record.getMessage()
        if "request_id" not in kv_fields:
            kv_fields["request_id"] = getattr(record, "request_id", "-")
        kv_fields["level"] = record.levelname
        kv_fields["logger"] = record.name

        pairs = [timestamp]
        for key, value in kv_fields.items():
            formatted = _format_value(value)
            pairs.append(f"{key}={formatted}")
        return " ".join(pairs)


def setup_logging(*, level: str = "INFO") -> None:
    """Configure root logging with the key=value formatter."""
    # Configure root logger
    root = logging.getLogger()
    root.handlers.clear()
    root_handler = logging.StreamHandler(sys.stdout)
    root_handler.addFilter(RequestIdFilter())
    root_handler.setFormatter(KeyValueFormatter())
    root.addHandler(root_handler)
    root.setLevel(level.upper())

    logging.captureWarnings(True)
    for noisy in ("httpx", "sqlalchemy.engine", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # IMPORTANT: Also add handlers directly to app loggers to survive other logging reconfigurations
    # This ensures our formatter works even if something else modifies the root logger
    for app_logger_name in ("app", "app.semantic_cache", "app.semantic_cache.repository", "app.cache", "app.proxy", "app.policy"):
        app_logger = logging.getLogger(app_logger_name)
        app_logger.setLevel(level.upper())
        # CRITICAL: Ensure the logger is not disabled
        app_logger.disabled = False
        # Clear any existing handlers on this logger
        app_logger.handlers.clear()
        # Add our custom handler directly to this logger
        app_handler = logging.StreamHandler(sys.stdout)
        app_handler.addFilter(RequestIdFilter())
        app_handler.setFormatter(KeyValueFormatter())
        app_handler.setLevel(logging.NOTSET)  # Let the logger level control filtering
        app_logger.addHandler(app_handler)
        # Don't propagate to root to avoid duplicate logs
        app_logger.propagate = False


def with_kv(event: str, **fields: Any) -> dict[str, Any]:
    """Prepare the kv payload for logging calls."""
    payload = {"event": event}
    payload.update(fields)
    return {"kv": payload}


def log_request_start(**fields: Any) -> None:
    """Log the beginning of a request lifecycle."""
    logging.getLogger("app.proxy").info(
        "proxy.request_start",
        extra=with_kv("proxy.request_start", **fields),
    )


def log_request_complete(**fields: Any) -> None:
    """Log the completion of a request lifecycle."""
    logging.getLogger("app.proxy").info(
        "proxy.request_complete",
        extra=with_kv("proxy.request_complete", **fields),
    )


def log_cache_event(event: str, **fields: Any) -> None:
    """Log semantic cache decisions and mutations."""
    logging.getLogger("app.cache").info(
        event,
        extra=with_kv(event, **fields),
    )


def log_policy_event(event: str, **fields: Any) -> None:
    """Log policy decisions."""
    logging.getLogger("app.policy").info(
        event,
        extra=with_kv(event, **fields),
    )


def _format_value(value: Any) -> str:
    """Render values safely for key=value output."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value).lower() if isinstance(value, bool) else str(value)
    if value is None:
        return "-"
    text = str(value)
    if any(ch.isspace() for ch in text) or "=" in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text
