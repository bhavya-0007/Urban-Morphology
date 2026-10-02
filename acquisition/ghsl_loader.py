"""GHSL built-up surface loader — historical footprint validation layer.

FIX (see chat): GHS_BUILT_S over the AOI was confirmed via reduceRegion
to range [0, 10000] -- m^2 of built-up floor area within each 100 m
pixel (100 x 100 = 10,000 m^2 at full build-out), NOT a [0,1] fraction
as urban_extent.py's `ghsl_threshold=0.2` assumed. That made the GHSL
vote in the 3-source majority fusion pass for nearly every pixel with
any built-up area at all. Dividing by the fixed pixel-area denominator
here (rather than tweaking the threshold in urban_extent.py) keeps the
band's semantics as a real [0,1] fraction, so every consumer's
thresholds stay comparable to ISF's [0,1] scale.

Also corrects config.datasets.GHSL_BUILT_SPEC.resolution_m, which was
declared as 10 but should be 100 (confirmed by the 10,000 max, i.e.
100m x 100m pixels) -- update that separately in datasets.py.
"""
from __future__ import annotations

from typing import Any

from config.datasets import GHSL_BUILT_SPEC
from config.settings import Settings
from core.ee_manager import EarthEngineManager
from core.logger import get_logger
from core.validation import validate_aoi

logger = get_logger(__name__)

# GHSL BUILT_S epochs available in the GHS-BUILT-S P2023A release.
GHSL_EPOCHS: list[int] = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]

# Confirmed via reduceRegion: native GHS_BUILT_S values are m^2 built-up
# area within a 100 m pixel (max = 10,000 = fully built 100m x 100m
# pixel). Dividing by this converts to a true [0,1] fraction.
GHSL_PIXEL_AREA_M2 = 10_000.0


class GHSLLoader:
    """Loader for JRC Global Human Settlement Layer built-up surface."""

    def __init__(self, settings: Settings, ee_manager: EarthEngineManager) -> None:
        self.settings = settings
        self.ee_manager = ee_manager

    @property
    def label(self) -> str:
        return "GHSL"

    def _nearest_epoch(self, year: int) -> int:
        return min(GHSL_EPOCHS, key=lambda epoch: abs(epoch - year))

    def load(self, year: int, aoi: Any | None = None) -> Any:
        """Return the built-up surface FRACTION image nearest to ``year``.

        Args:
            year: Requested study year.
            aoi: Optional ``ee.Geometry``; defaults to the city AOI.

        Returns:
            An ``ee.Image`` clipped to the AOI with a single band
            ``built_surface_fraction`` in ``[0, 1]``.
        """
        import ee

        self.ee_manager.ensure_initialized()
        aoi = aoi if aoi is not None else self.ee_manager.get_aoi_geometry()
        validate_aoi(aoi, f"{self.label} {year}")

        epoch = self._nearest_epoch(year)
        if epoch != year:
            logger.info("GHSL: no epoch for %d, using nearest epoch %d.", year, epoch)

        collection = ee.ImageCollection(GHSL_BUILT_SPEC.collection_id)
        image = (
            collection.filter(ee.Filter.eq("system:index", str(epoch)))
            .first()
        )
        image = ee.Image(image).select([0], ["built_surface_fraction"])
        # Convert raw m^2-built-per-pixel to a true [0,1] fraction.
        image = image.divide(GHSL_PIXEL_AREA_M2).clamp(0, 1).clip(aoi)
        image = image.set({"year": year, "epoch_used": epoch, "source_dataset": GHSL_BUILT_SPEC.collection_id})
        return image
