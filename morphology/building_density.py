"""Building density estimation from urban extent + optional footprints."""
from __future__ import annotations

from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


class BuildingDensityCalculator:

    def __init__(
        self,
        kernel_radius_m: int = 200,
    ):
        self.kernel_radius_m = kernel_radius_m

    def compute_from_isf(
        self,
        isf_image,
    ):
        import ee

        kernel = ee.Kernel.circle(
            radius=self.kernel_radius_m,
            units="meters",
        )

        density = (
            isf_image
            .select("ISF")
            .reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=kernel,
            )
            .rename("building_density")
        )

        return density.clamp(0, 1)

    def compute_from_urban_extent(
        self,
        urban_extent,
    ):
        import ee

        kernel = ee.Kernel.circle(
            radius=self.kernel_radius_m,
            units="meters",
        )

        density = (
            urban_extent
            .select("urban_extent")
            .reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=kernel,
            )
            .rename("building_density")
        )

        return density.clamp(0, 1)