"""GLC_FCS30D global land cover loader — used for impervious/urban fusion.

FIX (see chat): confirmed via debug_glcfcs_bands.py that this asset is
NOT one image per year. Each geographic tile is a single image with 23
bands (b1..b23), one band per year, spanning 2000-2022 inclusive
(id_no='GLC_FCS30D_20002022_...'). The original loader's
`.filterDate(...)` / `.filter(ee.Filter.eq("system:index", str(epoch)))`
logic was filtering an ImageCollection that only ever contains 1 image
for a given AOI -- it was never selecting a specific year, it was just
returning whatever band happened to be first. This version selects the
correct band directly via positional index.
"""
from __future__ import annotations

from typing import Any

from config.datasets import GLC_FCS30D_SPEC
from config.settings import Settings
from core.ee_manager import EarthEngineManager
from core.logger import get_logger
from core.validation import validate_aoi

logger = get_logger(__name__)

# GLC_FCS30D impervious-surface class code (30 m annual product).
IMPERVIOUS_CLASS_CODE = 190

# Confirmed via debug_glcfcs_bands.py: band b<N> corresponds to year
# (1999 + N), i.e. b1=2000 ... b23=2022. Update these if a future tile
# covers a different span -- read id_no/system:index to confirm before
# assuming this mapping holds for a new AOI/tile.
GLCFCS_FIRST_BAND_YEAR = 2000
GLCFCS_LAST_BAND_YEAR = 2022


class GLCFCSLoader:
    """Loader for the GLC_FCS30D annual land-cover classification."""

    def __init__(self, settings: Settings, ee_manager: EarthEngineManager) -> None:
        self.settings = settings
        self.ee_manager = ee_manager

    @property
    def label(self) -> str:
        return "GLC_FCS30D"

    def _band_for_year(self, year: int) -> tuple[str, int]:
        """Resolve the correct positional band name for a study year.

        Args:
            year: Requested study year.

        Returns:
            Tuple of ``(band_name, year_actually_used)``. If ``year``
            falls outside the tile's available span, clamps to the
            nearest available year (matching GHSLLoader's nearest-epoch
            behavior) rather than raising.
        """
        clamped_year = min(max(year, GLCFCS_FIRST_BAND_YEAR), GLCFCS_LAST_BAND_YEAR)
        if clamped_year != year:
            logger.info(
                "GLC_FCS30D: no band for %d (tile spans %d-%d), using nearest year %d.",
                year, GLCFCS_FIRST_BAND_YEAR, GLCFCS_LAST_BAND_YEAR, clamped_year,
            )
        band_index = clamped_year - GLCFCS_FIRST_BAND_YEAR + 1  # b1 = first year
        return f"b{band_index}", clamped_year

    def load(self, year: int, aoi: Any | None = None) -> Any:
        """Return a binary impervious-surface mask for the requested year.

        Args:
            year: Requested study year.
            aoi: Optional ``ee.Geometry``; defaults to the city AOI.

        Returns:
            An ``ee.Image`` with a single band ``impervious`` (1 = built-up
            impervious surface, 0 = other).
        """
        import ee

        self.ee_manager.ensure_initialized()
        aoi = aoi if aoi is not None else self.ee_manager.get_aoi_geometry()
        validate_aoi(aoi, f"{self.label} {year}")

        band_name, year_used = self._band_for_year(year)

        # mosaic() rather than first(): safe if a future AOI spans more
        # than one tile (this Hyderabad AOI currently sits in exactly
        # one tile, per debug_glcfcs_bands.py, but this keeps the loader
        # correct if that ever changes).
        collection = ee.ImageCollection(GLC_FCS30D_SPEC.collection_id).filterBounds(aoi)
        land_cover_band = collection.select([band_name]).mosaic()

        impervious = land_cover_band.eq(IMPERVIOUS_CLASS_CODE).rename("impervious")
        impervious = impervious.clip(aoi).set(
            {"year": year, "year_used": year_used, "band_used": band_name,
             "source_dataset": GLC_FCS30D_SPEC.collection_id}
        )
        return impervious