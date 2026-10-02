"""Normalized Difference Built-up Index."""
from __future__ import annotations

from typing import Any

from indices.base_index import BaseIndex


class NDBI(BaseIndex):
    """NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)."""

    name = "NDBI"
    required_bands = ["swir1", "nir"]

    def _formula(self, image: Any) -> Any:

        ndbi = image.normalizedDifference(
            ["swir1", "nir"]
        )

        # Reduce isolated noisy pixels
        ndbi = ndbi.focalMedian(
            radius=1,
            units="pixels"
        )

        return ndbi.rename("ndbi")