"""Generic, dependency-light utility functions reused across modules."""
from __future__ import annotations

import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def season_date_range(year: int, start_month_day: str, end_month_day: str) -> tuple[str, str]:
    """Build ISO start/end date strings for a seasonal window in a given year.

    Args:
        year: Calendar year.
        start_month_day: ``"MM-DD"`` start of season.
        end_month_day: ``"MM-DD"`` end of season.

    Returns:
        Tuple of ``(start_date, end_date)`` as ``"YYYY-MM-DD"`` strings.
    """
    return f"{year}-{start_month_day}", f"{year}-{end_month_day}"


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (including parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The same path as a :class:`Path`, guaranteed to exist.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename component.

    Args:
        name: Raw string, possibly containing spaces or punctuation.

    Returns:
        A lowercase string with only ``[a-z0-9_]`` characters.
    """
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def retry(
    times: int = 3, delay_seconds: float = 2.0, exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function on failure with linear backoff.

    Args:
        times: Maximum number of attempts.
        delay_seconds: Base delay between attempts; multiplied by attempt
            number for linear backoff.
        exceptions: Exception types that trigger a retry.

    Returns:
        A decorator wrapping the target function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "%s failed (attempt %d/%d): %s", func.__name__, attempt, times, exc
                    )
                    if attempt < times:
                        time.sleep(delay_seconds * attempt)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into fixed-size chunks.

    Args:
        items: The list to split.
        chunk_size: Maximum size of each chunk.

    Returns:
        A list of sub-lists, the last of which may be shorter.
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


class Timer:
    """Simple context manager for timing a block of code.

    Example:
        >>> with Timer("preprocessing") as t:
        ...     do_work()
        >>> print(t.elapsed_seconds)
    """

    def __init__(self, label: str = "block") -> None:
        self.label = label
        self.elapsed_seconds: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
        logger.info("%s completed in %.2fs", self.label, self.elapsed_seconds)
