"""Normalized Difference Vegetation Index."""
from __future__ import annotations

from typing import Any

from indices.base_index import BaseIndex


class NDVI(BaseIndex):
    """NDVI = (NIR - Red) / (NIR + Red).

    Standard vegetation greenness index; low/negative values indicate
    built-up or bare surfaces, high values indicate healthy vegetation.
    """

    name = "NDVI"
    required_bands = ["nir", "red"]

    def _formula(self, image: Any) -> Any:
        return image.normalizedDifference(["nir", "red"])
