"""Sentinel-1 GRD SAR loader for built-up structure/texture enhancement."""
from __future__ import annotations

from typing import Any

from acquisition.base import BaseLoader
from config.datasets import SENTINEL1_SPEC
from core.logger import get_logger

logger = get_logger(__name__)

SENTINEL1_MIN_YEAR = 2015


class Sentinel1Loader(BaseLoader):
    """Loader for Sentinel-1 GRD (VV/VH, IW mode, ascending+descending)."""

    @property
    def label(self) -> str:
        return "Sentinel-1"

    def _build_collection(self, year: int, aoi: Any) -> tuple[Any, list[str]]:
        import ee

        if year < SENTINEL1_MIN_YEAR:
            raise ValueError(
                f"Sentinel-1 has no usable coverage for year {year} "
                f"(available from {SENTINEL1_MIN_YEAR})."
            )

        start_date, end_date = self._date_range(year)
        collection = (
            ee.ImageCollection(SENTINEL1_SPEC.collection_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )
        return collection, [SENTINEL1_SPEC.collection_id]

    def _harmonize_bands(self, collection: Any) -> Any:
        native_to_canonical = {v: k for k, v in SENTINEL1_SPEC.bands.items()}

        def _mapper(image: Any) -> Any:
            # GRD backscatter is already in dB; apply a light speckle
            # smoothing filter before renaming.
            smoothed = image.focalMedian(radius=30, units="meters")
            renamed = smoothed.select(list(native_to_canonical.keys())).rename(
                list(native_to_canonical.values())
            )
            return renamed.copyProperties(image, image.propertyNames())

        return collection.map(_mapper)
