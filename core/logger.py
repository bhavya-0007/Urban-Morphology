"""Centralized logging configuration.

All modules must obtain loggers via :func:`get_logger` instead of calling
``print``. This gives consistent formatting, log-level control via
``Settings.log_level``, and optional file logging under
``outputs/logs/``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED_LOGGERS: set[str] = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: Path | None = None,
) -> logging.Logger:
    """Return a configured logger, memoized by name.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Logging level string, e.g. ``"DEBUG"``, ``"INFO"``.
        log_dir: If provided, also write logs to ``{log_dir}/pipeline.log``.

    Returns:
        A ``logging.Logger`` with a console handler (and optional file
        handler) attached exactly once.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level.upper())
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger
