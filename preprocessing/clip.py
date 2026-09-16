"""AOI clipping and reprojection helpers."""
from __future__ import annotations

from typing import Any

from core.logger import get_logger
from core.validation import validate_aoi, validate_crs

logger = get_logger(__name__)


def clip_to_aoi(image: Any, aoi: Any, label: str = "image") -> Any:
    """Clip an image to an AOI, validating the geometry first.

    Args:
        image: An ``ee.Image``.
        aoi: ``ee.Geometry`` AOI.
        label: Label used in validation error messages.

    Returns:
        The clipped ``ee.Image``.
    """
    validate_aoi(aoi, label)
    return image.clip(aoi)


def reproject_image(image: Any, crs: str, scale: int, label: str = "image") -> Any:
    """Reproject an image to the project CRS at a fixed scale.

    Args:
        image: An ``ee.Image``.
        crs: Target CRS, e.g. ``"EPSG:32644"``.
        scale: Target pixel size in meters.
        label: Label used in validation error messages.

    Returns:
        The reprojected ``ee.Image``.
    """
    validate_crs(crs, label)
    return image.reproject(crs=crs, scale=scale)


def buffer_aoi(aoi: Any, buffer_meters: float) -> Any:
    """Buffer an AOI outward, useful to avoid edge artifacts in texture/
    landscape-metric computations that require a neighborhood context.

    Args:
        aoi: ``ee.Geometry`` AOI.
        buffer_meters: Buffer distance in meters (can be negative to
            shrink).

    Returns:
        The buffered ``ee.Geometry``.
    """
    return aoi.buffer(buffer_meters)
