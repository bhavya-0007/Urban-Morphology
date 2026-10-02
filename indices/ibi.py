"""
Index-based Built-up Index (IBI)

Xu (2008)

IBI = (NDBI' - (SAVI' + MNDWI') / 2)
      --------------------------------
      (NDBI' + (SAVI' + MNDWI') / 2)

where each sub-index is normalized using
scene-specific 2nd and 98th percentiles.
"""

from __future__ import annotations

from typing import Any

from indices.base_index import BaseIndex
from indices.ndbi import NDBI
from indices.mndwi import MNDWI


class IBI(BaseIndex):

    name = "IBI"

    required_bands = [
        "nir",
        "red",
        "green",
        "swir1",
    ]

    savi_l = 0.5
    stats_scale_m = 60

    def _formula(self, image: Any) -> Any:

        import ee

        # ---------------------------------------------------------
        # Component indices
        # ---------------------------------------------------------

        ndbi = (
            NDBI()
            ._formula(image)
            .rename("ndbi")
        )

        mndwi = (
            MNDWI()
            ._formula(image)
            .rename("mndwi")
        )

        savi = image.expression(
            "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
            {
                "NIR": image.select("nir"),
                "RED": image.select("red"),
                "L": self.savi_l,
            },
        ).rename("savi")

        # ---------------------------------------------------------
        # Remove invalid pixels
        # ---------------------------------------------------------

        valid_mask = (
            ndbi.mask()
            .And(mndwi.mask())
            .And(savi.mask())
        )

        stack = (
            ee.Image.cat([
                ndbi,
                mndwi,
                savi,
            ])
            .updateMask(valid_mask)
        )

        region = image.geometry()

        # ---------------------------------------------------------
        # Percentile statistics
        # ---------------------------------------------------------

        stats = stack.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=region,
            scale=self.stats_scale_m,
            bestEffort=True,
            maxPixels=1e13,
            tileScale=8,
        )

        # ---------------------------------------------------------
        # Safe normalization
        # ---------------------------------------------------------

        def normalize(img, band):

            lo = ee.Number(
                ee.Algorithms.If(
                    stats.get(f"{band}_p2"),
                    stats.get(f"{band}_p2"),
                    -1,
                )
            )

            hi = ee.Number(
                ee.Algorithms.If(
                    stats.get(f"{band}_p98"),
                    stats.get(f"{band}_p98"),
                    1,
                )
            )

            hi = ee.Number(
                ee.Algorithms.If(
                    hi.subtract(lo).abs().lt(1e-6),
                    lo.add(1),
                    hi,
                )
            )

            return (
                img
                .unitScale(lo, hi)
                .clamp(0, 1)
            )

        ndbi_n = normalize(ndbi, "ndbi")
        mndwi_n = normalize(mndwi, "mndwi")
        savi_n = normalize(savi, "savi")

        # ---------------------------------------------------------
        # Xu (2008) IBI
        # ---------------------------------------------------------

        veg_water = (
            savi_n
            .add(mndwi_n)
            .divide(2)
        )

        numerator = ndbi_n.subtract(veg_water)

        denominator = ndbi_n.add(veg_water)

        denominator = denominator.where(
            denominator.abs().lt(1e-6),
            1e-6,
        )

        ibi = (
            numerator
            .divide(denominator)
            .rename("ibi")
        )

        return ibi