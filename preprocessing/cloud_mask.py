"""Generic, sensor-agnostic cloud masking helpers.

Sensor-specific bitmask logic lives inside each acquisition loader
(where the QA band semantics differ); this module provides shared
post-hoc mask utilities used by later pipeline stages, such as
re-checking valid-pixel coverage after masking.
"""
from __future__ import annotations

from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


def valid_pixel_fraction(image: Any, region: Any, scale: int) -> float:
    """Compute the fraction of unmasked pixels within a region.

    Args:
        image: A masked ``ee.Image``.
        region: ``ee.Geometry`` region to evaluate.
        scale: Pixel scale in meters for the reduction.

    Returns:
        Fraction of valid (unmasked) pixels in ``[0, 1]``.
    """
    import ee

    band = image.bandNames().get(0)
    mask = image.select([band]).mask()
    stats = mask.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e13
    )
    value = stats.getInfo().get(band.getInfo())
    return float(value) if value is not None else 0.0


def fill_gaps_with_temporal_median(
    collection: Any, target_image: Any, mask_band: str
) -> Any:
    """Fill masked gaps in ``target_image`` using the collection's median.

    Args:
        collection: The full ``ee.ImageCollection`` the target was drawn
            from (or composited from).
        target_image: The image with gaps to fill.
        mask_band: Band name used to determine gap locations.

    Returns:
        A gap-filled ``ee.Image``.
    """
    fallback = collection.median()
    gap_mask = target_image.select(mask_band).mask().Not()
    filled = target_image.unmask(0).where(gap_mask, fallback)
    return filled.copyProperties(target_image, target_image.propertyNames())
