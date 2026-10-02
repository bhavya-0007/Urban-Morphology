from __future__ import annotations

from typing import Any

from indices.base_index import BaseIndex
from indices.ibi import IBI


class ISF(BaseIndex):
    """ISF: rescaled IBI clipped to ``[0, 1]``, representing the estimated
    fraction of each pixel occupied by impervious/built-up surface.

    This is a continuous proxy used as an ML feature; it is later fused
    with the GLC_FCS30D binary impervious mask during morphology
    extraction for a more robust urban-extent product.
    """

    name = "ISF"
    required_bands = ["nir", "red", "swir1", "green"]

    def _formula(self, image: Any) -> Any:
        import ee

        ibi = IBI()._formula(image)

        region = image.geometry()

        stats = ibi.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=region,
            scale=30,
            bestEffort=True,
            maxPixels=1e13,
            tileScale=4,
        )

        lo = ee.Number(stats.get("ibi_p2"))
        hi = ee.Number(stats.get("ibi_p98"))

        lo = ee.Number(ee.Algorithms.If(lo, lo, -1))
        hi = ee.Number(ee.Algorithms.If(hi, hi, 1))

        hi = ee.Number(
            ee.Algorithms.If(
                hi.subtract(lo).abs().lt(1e-6),
                lo.add(1),
                hi,
            )
        )

        isf = (
            ibi
            .unitScale(lo, hi)
            .clamp(0, 1)
            .focalMedian(radius=1, units="pixels")
            .rename("isf")
        )

        return isf


