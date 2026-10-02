"""Abstract base class for all spectral index calculators.

Per the PDD, each spectral index is implemented as its own class exposing
a ``compute(image) -> ee.Image`` method returning a single-band image
named after the index (e.g. ``"NDVI"``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.logger import get_logger
from core.validation import validate_bands

logger = get_logger(__name__)


class BaseIndex(ABC):
    """Common contract for spectral index calculators."""

    #: Output band name for this index.
    name: str = "INDEX"
    #: Canonical band names required as input.
    required_bands: list[str] = []

    def compute(self, image: Any) -> Any:
        """Validate inputs and compute the index band.

        Args:
            image: An ``ee.Image`` with canonical band names (blue, green,
                red, nir, swir1, swir2, ...).

        Returns:
            A single-band ``ee.Image`` named ``self.name``.
        """
        validate_bands(image, self.required_bands, self.name)
        import ee
        result = (
            ee.Image(self._formula(image))
            .rename(self.name)
            .toFloat()
            )
        result = ee.Image(
            result.copyProperties(image, image.propertyNames())
            )
        return result

    @abstractmethod
    def _formula(self, image: Any) -> Any:
        """Band-math implementation, without renaming or validation."""
