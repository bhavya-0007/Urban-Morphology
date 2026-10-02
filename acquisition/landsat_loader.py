"""Landsat Collection 2 Level-2 surface reflectance loader.

Handles TM (LT05), ETM+ (LE07), and OLI (LC08/LC09) sensors, applying the
correct scale/offset per sensor family and a shared bitmask cloud filter.
"""
from __future__ import annotations

from typing import Any

from acquisition.base import BaseLoader
from config.datasets import LANDSAT_COLLECTIONS, LANDSAT_YEAR_MAP
from core.logger import get_logger

logger = get_logger(__name__)


class LandsatLoader(BaseLoader):
    """Loader for multi-sensor Landsat Collection 2 Level-2 imagery."""

    @property
    def label(self) -> str:
        return "Landsat"

    def _sensors_for_year(self, year: int) -> list[str]:
        try:
            return LANDSAT_YEAR_MAP[year]
        except KeyError as exc:
            raise ValueError(
                f"No Landsat sensor mapping configured for year {year}."
            ) from exc

    def _build_collection(self, year: int, aoi: Any) -> tuple[Any, list[str]]:
        import ee

        start_date, end_date = self._date_range(year)
        sensor_keys = self._sensors_for_year(year)

        merged: Any | None = None
        source_ids: list[str] = []

        for sensor_key in sensor_keys:
            spec = LANDSAT_COLLECTIONS[sensor_key]
            source_ids.append(spec.collection_id)
            coll = (
                ee.ImageCollection(spec.collection_id)
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUD_COVER", self.max_cloud_cover))
                .map(self._mask_clouds)
                .map(lambda img, sk=sensor_key: img.set("sensor_key", sk))
            )
            merged = coll if merged is None else merged.merge(coll)

        assert merged is not None
        return merged, source_ids

    @staticmethod
    def _mask_clouds(image: Any) -> Any:
        """Apply the QA_PIXEL bitmask to remove cloud/shadow/snow pixels."""
        qa = image.select("QA_PIXEL")
        cloud_shadow_bit = 1 << 4
        cloud_bit = 1 << 3
        cirrus_bit = 1 << 2
        snow_bit = 1 << 5
        mask = (
            qa.bitwiseAnd(cloud_shadow_bit).eq(0)
            .And(qa.bitwiseAnd(cloud_bit).eq(0))
            .And(qa.bitwiseAnd(cirrus_bit).eq(0))
            .And(qa.bitwiseAnd(snow_bit).eq(0))
        )
        return image.updateMask(mask)

    def _harmonize_bands(self, collection: Any) -> Any:
        """Rename/scale bands per-sensor, dispatching on ``sensor_key``.

        Earth Engine cannot branch server-side inside a single ``.map()``
        across differing band layouts, so each sensor's sub-collection
        (tagged during ``_build_collection``) is filtered out, mapped with
        its own native-band mapping, and the harmonized subsets are
        re-merged.
        """
        import ee

        def make_mapper(spec_bands: dict[str, str], scale: float, offset: float) -> Any:
            native_to_canonical = {v: k for k, v in spec_bands.items()}
            native_bands = list(native_to_canonical.keys())
            canonical_bands = [native_to_canonical[b] for b in native_bands]

            def _mapper(image: Any) -> Any:
                optical = (
                    image.select(native_bands)
                    .multiply(scale)
                    .add(offset)
                    .rename(canonical_bands)
                )
                return optical.copyProperties(image, image.propertyNames())

            return _mapper

        harmonized_subsets = []
        for sensor_key, spec in LANDSAT_COLLECTIONS.items():
            subset = collection.filter(ee.Filter.eq("sensor_key", sensor_key))
            mapper = make_mapper(spec.bands, spec.scale_factor, spec.offset)
            harmonized_subsets.append(subset.map(mapper))

        merged = harmonized_subsets[0]
        for subset in harmonized_subsets[1:]:
            merged = merged.merge(subset)
        return merged
