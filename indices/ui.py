"""Urban Index."""
from __future__ import annotations

from typing import Any

from indices.base_index import BaseIndex


class UI(BaseIndex):
    """UI = (SWIR2 - NIR) / (SWIR2 + NIR).

    Simple built-up enhancement index using the far-SWIR band, effective
    at separating urban surfaces from vegetation in arid/semi-arid
    settings such as Hyderabad's pre-monsoon composites.
    """

    name = "UI"
    required_bands = ["swir2", "nir"]

    def _formula(self, image: Any) -> Any:
        return image.normalizedDifference(["swir2", "nir"])
