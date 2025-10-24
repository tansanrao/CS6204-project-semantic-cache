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
    root = logging.getLogger()
    preserved = [
        handler
        for handler in root.handlers
        if handler.__class__.__module__.startswith("_pytest")
        or handler.__class__.__name__ == "LogCaptureHandler"
    ]
    for handler in list(root.handlers):
        if handler not in preserved:
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(KeyValueFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    logging.captureWarnings(True)
    for noisy in ("httpx", "sqlalchemy.engine", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
