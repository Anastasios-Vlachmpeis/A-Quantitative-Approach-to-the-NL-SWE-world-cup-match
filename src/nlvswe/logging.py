"""Logging setup for nlvswe."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger with consistent timestamp + level formatting."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root = logging.getLogger("nlvswe")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
