"""RGB preview / quicklook generation for QA of composites."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


def generate_rgb_thumbnail_url(
    image: Any,
    region: Any,
    bands: tuple[str, str, str] = ("red", "green", "blue"),
    min_value: float = 0.0,
    max_value: float = 0.3,
    dimensions: int = 768,
) -> str:
    """Build a quicklook thumbnail URL for visual QA of a composite.

    Args:
        image: An ``ee.Image`` with canonical band names.
        region: ``ee.Geometry`` to render.
        bands: Bands mapped to (R, G, B).
        min_value: Display stretch minimum.
        max_value: Display stretch maximum.
        dimensions: Max dimension (pixels) of the returned thumbnail.

    Returns:
        A signed URL string that renders the PNG thumbnail.
    """
    vis_params = {
        "bands": list(bands),
        "min": min_value,
        "max": max_value,
        "region": region,
        "dimensions": dimensions,
        "format": "png",
    }
    return image.getThumbURL(vis_params)


def save_thumbnail(url: str, output_path: Path) -> Path:
    """Download a thumbnail URL to local disk.

    Args:
        url: URL returned by :func:`generate_rgb_thumbnail_url`.
        output_path: Destination PNG path.

    Returns:
        The path written to.
    """
    import urllib.request

    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)  # noqa: S310 - trusted EE URL
    logger.info("Saved preview thumbnail: %s", output_path)
    return output_path
