"""Sentinel-2 L2A surface reflectance loader (10 m morphology detail)."""
from __future__ import annotations

from typing import Any

from acquisition.base import BaseLoader
from config.datasets import SENTINEL2_SPEC
from core.logger import get_logger

logger = get_logger(__name__)

# Sentinel-2 data is only reliably available from mid-2015 onward.
SENTINEL2_MIN_YEAR = 2017


class Sentinel2Loader(BaseLoader):
    """Loader for harmonized Sentinel-2 Level-2A imagery."""

    @property
    def label(self) -> str:
        return "Sentinel-2"

    def _build_collection(self, year: int, aoi: Any) -> tuple[Any, list[str]]:
        import ee

        if year < SENTINEL2_MIN_YEAR:
            raise ValueError(
                f"Sentinel-2 has no usable coverage for year {year} "
                f"(available from {SENTINEL2_MIN_YEAR})."
            )

        start_date, end_date = self._date_range(year)
        collection = (
            ee.ImageCollection(SENTINEL2_SPEC.collection_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.max_cloud_cover))
            .map(self._mask_clouds)
        )
        return collection, [SENTINEL2_SPEC.collection_id]

    @staticmethod
    def _mask_clouds(image: Any) -> Any:
        """Mask clouds/cirrus using the QA60 bitmask and SCL classes."""
        qa60 = image.select("QA60")
        cloud_bit = 1 << 10
        cirrus_bit = 1 << 11
        qa_mask = qa60.bitwiseAnd(cloud_bit).eq(0).And(qa60.bitwiseAnd(cirrus_bit).eq(0))

        scl = image.select("SCL")
        # Exclude SCL classes: 3 (cloud shadow), 8/9 (cloud medium/high
        # probability), 10 (thin cirrus), 11 (snow/ice).
        scl_mask = scl.remap([3, 8, 9, 10, 11], [0, 0, 0, 0, 0], 1)

        return image.updateMask(qa_mask).updateMask(scl_mask)

    def _harmonize_bands(self, collection: Any) -> Any:
        native_to_canonical = {v: k for k, v in SENTINEL2_SPEC.bands.items()}
        native_bands = list(native_to_canonical.keys())
        canonical_bands = [native_to_canonical[b] for b in native_bands]
        scale = SENTINEL2_SPEC.scale_factor

        def _mapper(image: Any) -> Any:
            optical = image.select(native_bands).multiply(scale).rename(canonical_bands)
            return optical.copyProperties(image, image.propertyNames())

        return collection.map(_mapper)
