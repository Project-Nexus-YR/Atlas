"""Logging setup.

Atlas prints exactly one thing to stdout - the results table. Everything else
goes through :mod:`logging` to stderr, so a run can be piped without the
narration contaminating the numbers.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Attach a stderr handler once. Safe to call from any entry point."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)-22s %(message)s"))
    root = logging.getLogger("atlas")
    root.setLevel(level.upper())
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger."""
    return logging.getLogger(f"atlas.{name}")
