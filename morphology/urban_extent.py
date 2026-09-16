"""Urban extent delineation via multi-source fusion.

Combines the continuous ISF spectral index, the GHSL built-up fraction,
and the GLC_FCS30D binary impervious mask into a single validated urban
extent layer, reducing false positives/negatives from any one source.
"""

from __future__ import annotations

from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


class UrbanExtentExtractor:
    """Fuses spectral, GHSL, and land-cover sources into an urban mask."""

    def __init__(
        self,
        isf_threshold: float = 0.40,
        ghsl_threshold: float = 0.25,
        min_agreement: int = 2,
        morphology_radius: int = 1,
    ) -> None:

        self.isf_threshold = isf_threshold
        self.ghsl_threshold = ghsl_threshold
        self.min_agreement = min_agreement
        self.morphology_radius = morphology_radius

    def extract(
        self,
        isf_image: Any,
        ghsl_image: Any,
        glcfcs_impervious: Any,
        mndwi_image: Any | None = None,
        water_threshold: float = 0.0,
    ) -> Any:
        """
        Produce the fused urban extent layer.
        """

        # -----------------------------
        # Individual source votes
        # -----------------------------
        vote_isf = isf_image.gte(self.isf_threshold)

        vote_ghsl = ghsl_image.gte(self.ghsl_threshold)

        vote_glcfcs = glcfcs_impervious.gte(1)

        # -----------------------------
        # Majority voting
        # -----------------------------
        agreement = (
            vote_isf
            .add(vote_ghsl)
            .add(vote_glcfcs)
        )

        urban = agreement.gte(
            self.min_agreement
        )

        # -----------------------------
        # Remove water pixels
        # -----------------------------
        if mndwi_image is not None:

            water_mask = (
                mndwi_image
                .select("MNDWI")
                .gt(water_threshold)
            )

            urban = urban.where(
                water_mask,
                0
            )

        # -----------------------------
        # Morphological cleaning
        #
        # Closing:
        #   focalMax -> fills tiny holes
        #   focalMin -> removes isolated pixels
        # -----------------------------
        urban = (
            urban
            .focalMax(
                radius=self.morphology_radius,
                units="pixels"
            )
            .focalMin(
                radius=self.morphology_radius,
                units="pixels"
            )
        )

        return (
            urban
            .selfMask()
            .unmask(0)
            .rename("urban_extent")
        )

    def urban_area_hectares(
        self,
        urban_extent: Any,
        region: Any,
        scale: int,
    ) -> float:
        """
        Compute total urban area in hectares.
        """

        import ee

        pixel_area = ee.Image.pixelArea()

        area_image = urban_extent.multiply(pixel_area)

        stats = area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
        )

        band = urban_extent.bandNames().get(0).getInfo()

        square_meters = float(
            stats.getInfo().get(
                band,
                0.0,
            )
            or 0.0
        )

        return square_meters / 10000.0