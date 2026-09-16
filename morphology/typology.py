"""Morphological typology classification.

Distinct from LCZ (a climate-oriented standard taxonomy), this module
derives a simpler descriptive typology (dense urban core, mixed
built-up, sparse/peri-urban, vegetated/open space, water, bare land)
directly from spectral indices and building density, primarily for
cartographic communication and as an ML target label.
"""
from __future__ import annotations

from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)

# Codes must match config.constants.MORPHOLOGY_CLASSES.
DENSE_URBAN_CORE = 0
MIXED_BUILT_UP = 1
SPARSE_PERI_URBAN = 2
VEGETATED_OPEN_SPACE = 3
WATER = 4
BARE_TRANSITIONAL = 5


class MorphologyTypologyClassifier:
    """Rule-based morphological typology using thresholded spectral bands."""

    def __init__(
        self,
        dense_density_threshold: float = 0.6,
        mixed_density_threshold: float = 0.3,
        sparse_density_threshold: float = 0.1,
        vegetation_ndvi_threshold: float = 0.4,
        water_mndwi_threshold: float = 0.0,
        bare_bsi_threshold: float = 0.2,
    ) -> None:
        self.dense_density_threshold = dense_density_threshold
        self.mixed_density_threshold = mixed_density_threshold
        self.sparse_density_threshold = sparse_density_threshold
        self.vegetation_ndvi_threshold = vegetation_ndvi_threshold
        self.water_mndwi_threshold = water_mndwi_threshold
        self.bare_bsi_threshold = bare_bsi_threshold

    def classify(
        self,
        building_density: Any,
        ndvi: Any,
        mndwi: Any,
        bsi: Any,
    ) -> Any:
        """Classify each pixel into a morphological typology code.

        Precedence: water > vegetation > bare land > building-density
        tiers. Earlier ``.where()`` calls are overwritten by later,
        higher-precedence ones so the final call wins ties.

        Args:
            building_density: Single-band ``ee.Image`` from
                :class:`morphology.building_density.BuildingDensityCalculator`.
            ndvi: Single-band NDVI ``ee.Image``.
            mndwi: Single-band MNDWI ``ee.Image``.
            bsi: Single-band BSI ``ee.Image``.

        Returns:
            A single-band ``ee.Image`` named ``"morphology_class"`` with
            integer codes matching
            :data:`config.constants.MORPHOLOGY_CLASSES`.
        """
        import ee

        density = building_density.select("building_density")

        result = ee.Image(SPARSE_PERI_URBAN).rename("morphology_class")
        result = result.where(density.gte(self.sparse_density_threshold), SPARSE_PERI_URBAN)
        result = result.where(density.gte(self.mixed_density_threshold), MIXED_BUILT_UP)
        result = result.where(density.gte(self.dense_density_threshold), DENSE_URBAN_CORE)

        bare_mask = bsi.select("BSI").gte(self.bare_bsi_threshold).And(density.lt(self.sparse_density_threshold))
        result = result.where(bare_mask, BARE_TRANSITIONAL)

        veg_mask = ndvi.select("NDVI").gte(self.vegetation_ndvi_threshold)
        result = result.where(veg_mask, VEGETATED_OPEN_SPACE)

        water_mask = mndwi.select("MNDWI").gte(self.water_mndwi_threshold)
        result = result.where(water_mask, WATER)

        return result.rename("morphology_class").toInt()
