"""Structured (key=value) logging configured once per process."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


class _KeyValueFormatter(logging.Formatter):
    """Render ``extra={"fields": {...}}`` as trailing ``key=value`` pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = getattr(record, "fields", None)
        if fields:
            pairs = " ".join(f"{k}={v}" for k, v in fields.items())
            base = f"{base} | {pairs}"
        return base


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger; configures the root handler on first call."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_KeyValueFormatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        root = logging.getLogger("titanic")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    short = name.removeprefix("src.")
    return logging.getLogger(f"titanic.{short}")
