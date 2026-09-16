"""Seasonal compositing strategies.

The default strategy used by :class:`acquisition.base.BaseLoader` is a
per-pixel median composite. This module exposes that (and alternatives)
as standalone, testable functions so compositing logic can be swapped or
compared without touching the loaders.
"""

from __future__ import annotations

from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


def median_composite(collection: Any, aoi: Any) -> Any:
    """Per-pixel median composite, robust to residual cloud contamination."""
    return collection.median().clip(aoi)


def urban_composite(collection: Any, aoi: Any) -> Any:
    """Urban-optimized composite.

    Selects, for every pixel, the observation having the
    highest NDBI. This preserves impervious surfaces better
    than a median composite.
    """

    def _add_ndbi(image: Any) -> Any:
        ndbi = (
            image.normalizedDifference(
                ["swir1", "nir"]
            )
            .rename("__urban_rank")
        )

        return image.addBands(ndbi)

    ranked = collection.map(_add_ndbi)

    composite = ranked.qualityMosaic("__urban_rank")

    return (
        composite
        .select(
            composite.bandNames().remove("__urban_rank")
        )
        .clip(aoi)
    )


def greenest_pixel_composite(
    collection: Any,
    aoi: Any,
    ndvi_band_source: tuple[str, str] = ("nir", "red"),
) -> Any:
    """Composite using the greenest (highest NDVI) valid pixel."""

    nir_band, red_band = ndvi_band_source

    def _add_ndvi(image: Any) -> Any:
        ndvi = image.normalizedDifference(
            [nir_band, red_band]
        ).rename("__ndvi_rank")

        return image.addBands(ndvi)

    ranked = collection.map(_add_ndvi)

    composite = ranked.qualityMosaic("__ndvi_rank")

    return (
        composite
        .select(
            composite.bandNames().remove("__ndvi_rank")
        )
        .clip(aoi)
    )


def mean_composite(collection: Any, aoi: Any) -> Any:
    """Per-pixel mean composite."""
    return collection.mean().clip(aoi)