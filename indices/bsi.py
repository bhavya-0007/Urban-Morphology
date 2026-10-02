"""Bare Soil Index."""
from __future__ import annotations

from typing import Any

from indices.base_index import BaseIndex


class BSI(BaseIndex):

    name = "BSI"

    required_bands = [
        "swir1",
        "red",
        "nir",
        "blue",
    ]

    def _formula(self, image: Any) -> Any:

        bsi = image.expression(
            "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
            {
                "SWIR1": image.select("swir1"),
                "RED": image.select("red"),
                "NIR": image.select("nir"),
                "BLUE": image.select("blue"),
            },
        )

        bsi = bsi.focalMedian(
            radius=1,
            units="pixels"
        )

        return bsi.rename("bsi")