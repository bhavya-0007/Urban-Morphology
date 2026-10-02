"""Modified Normalized Difference Water Index."""
from __future__ import annotations

from typing import Any

from indices.base_index import BaseIndex


class MNDWI(BaseIndex):
    """MNDWI = (Green - SWIR1) / (Green + SWIR1).

    Used to mask open water prior to urban morphology classification, so
    water bodies are not misclassified as built-up or bare land.
    """

    name = "MNDWI"
    required_bands = ["green", "swir1"]

    def _formula(self, image: Any) -> Any:
        return image.normalizedDifference(["green", "swir1"])
