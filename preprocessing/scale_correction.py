"""Reflectance scale correction and cross-sensor normalization utilities.

Per-sensor scale/offset application already happens inside the
acquisition loaders (using ``config.datasets`` specs). This module holds
additional, optional cross-sensor harmonization used when a single study
year mixes multiple sensors (e.g. LC08 + LC09 in 2025).
"""
from __future__ import annotations

from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


def clip_reflectance_range(image: Any, bands: list[str], min_value: float = 0.0, max_value: float = 1.0) -> Any:
    """Clamp reflectance bands to a physically valid range.

    Args:
        image: An ``ee.Image`` with scaled reflectance bands.
        bands: Band names to clamp.
        min_value: Minimum valid reflectance (default ``0.0``).
        max_value: Maximum valid reflectance (default ``1.0``).

    Returns:
        The image with clamped bands, other bands untouched.
    """
    clamped = image.select(bands).clamp(min_value, max_value)
    other_bands = image.bandNames().removeAll(bands)
    return image.select(other_bands).addBands(clamped).select(image.bandNames())


def apply_gain_offset_normalization(
    image: Any, band: str, gain: float, offset: float
) -> Any:
    """Apply a linear gain/offset correction to harmonize sensor drift.

    Useful for reconciling reflectance differences between e.g. Landsat 7
    ETM+ and Landsat 8 OLI over the same nominal band.

    Args:
        image: An ``ee.Image``.
        band: Band name to correct.
        gain: Multiplicative gain factor.
        offset: Additive offset.

    Returns:
        The image with the corrected band replacing the original.
    """
    corrected = image.select(band).multiply(gain).add(offset).rename(band)
    return image.addBands(corrected, overwrite=True)
