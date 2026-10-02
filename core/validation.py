"""Validation helpers shared across acquisition, preprocessing, and export.

Each function raises a specific subclass of
:class:`core.exceptions.UrbanMorphologyError` on failure so callers can
handle (or deliberately not handle) individual failure modes.
"""
from __future__ import annotations

from typing import Any

from core.exceptions import (
    EmptyCollectionError,
    InsufficientScenesError,
    InvalidCRSError,
    MissingAOIError,
    MissingBandError,
    ValidationError,
)
from core.logger import get_logger

logger = get_logger(__name__)


def validate_collection_size(collection: Any, min_scenes: int, label: str) -> int:
    """Ensure an ``ee.ImageCollection`` has at least ``min_scenes`` images.

    Args:
        collection: An ``ee.ImageCollection``.
        min_scenes: Minimum acceptable scene count.
        label: Human-readable label used in error messages (e.g.
            ``"Landsat 2010"``).

    Returns:
        The actual scene count.

    Raises:
        EmptyCollectionError: If the collection has zero scenes.
        InsufficientScenesError: If the collection has fewer than
            ``min_scenes`` scenes.
    """
    size = int(collection.size().getInfo())
    if size == 0:
        raise EmptyCollectionError(f"{label}: image collection is empty after filtering.")
    if size < min_scenes:
        raise InsufficientScenesError(
            f"{label}: only {size} scene(s) available, need at least {min_scenes}."
        )
    logger.info("%s: %d scene(s) passed filtering.", label, size)
    return size


def validate_bands(image: Any, required_bands: list[str], label: str) -> None:
    """Ensure an ``ee.Image`` contains all required bands.

    Args:
        image: An ``ee.Image``.
        required_bands: Band names that must be present.
        label: Human-readable label for error messages.

    Raises:
        MissingBandError: If any required band is absent.
    """
    band_names = set(image.bandNames().getInfo())
    missing = [b for b in required_bands if b not in band_names]
    if missing:
        raise MissingBandError(f"{label}: missing required band(s): {missing}")


def validate_crs(crs: str | None, label: str) -> str:
    """Ensure a CRS string is present and well-formed.

    Args:
        crs: CRS string, e.g. ``"EPSG:32644"``.
        label: Human-readable label for error messages.

    Returns:
        The validated CRS string.

    Raises:
        InvalidCRSError: If ``crs`` is missing or malformed.
    """
    if not crs or ":" not in crs:
        raise InvalidCRSError(f"{label}: invalid or missing CRS '{crs}'.")
    authority, code = crs.split(":", 1)
    if authority.upper() != "EPSG" or not code.isdigit():
        raise InvalidCRSError(f"{label}: unrecognized CRS format '{crs}'.")
    return crs


def validate_aoi(geometry: Any, label: str) -> None:
    """Ensure an AOI geometry is present and non-degenerate.

    Args:
        geometry: An ``ee.Geometry`` (or ``None``).
        label: Human-readable label for error messages.

    Raises:
        MissingAOIError: If ``geometry`` is ``None`` or has zero area.
    """
    if geometry is None:
        raise MissingAOIError(f"{label}: no AOI geometry supplied.")
    try:
        area = geometry.area(1).getInfo()
    except Exception as exc:  # noqa: BLE001
        raise MissingAOIError(f"{label}: could not evaluate AOI geometry: {exc}") from exc
    if area <= 0:
        raise MissingAOIError(f"{label}: AOI geometry has zero area.")


def validate_dataframe_not_empty(df: Any, label: str) -> None:
    """Ensure a pandas/geopandas DataFrame has at least one row.

    Args:
        df: A DataFrame-like object with ``len()`` support.
        label: Human-readable label for error messages.

    Raises:
        ValidationError: If the DataFrame is empty.
    """
    if len(df) == 0:
        raise ValidationError(f"{label}: resulting DataFrame is empty.")


def validate_value_range(
    value: float, min_value: float, max_value: float, label: str
) -> None:
    """Ensure a scalar falls within an expected range.

    Args:
        value: The value to check.
        min_value: Inclusive lower bound.
        max_value: Inclusive upper bound.
        label: Human-readable label for error messages.

    Raises:
        ValidationError: If ``value`` is outside ``[min_value, max_value]``.
    """
    if not (min_value <= value <= max_value):
        raise ValidationError(
            f"{label}: value {value} outside expected range [{min_value}, {max_value}]."
        )
