"""Abstract base class for all Earth Engine data loaders.

Every loader (Landsat, Sentinel-2, Sentinel-1, GHSL, GLC_FCS30D) follows
the same contract: given a year and AOI, return a validated, composited
``ee.Image`` with canonical band names.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from config.constants import MAX_CLOUD_COVER_PERCENT, MIN_SCENES_REQUIRED
from config.settings import Settings
from core.ee_manager import EarthEngineManager
from core.logger import get_logger
from core.utils import season_date_range
from core.validation import validate_aoi, validate_collection_size

logger = get_logger(__name__)


@dataclass
class LoaderResult:
    """Container for a loader's output image plus provenance.

    Attributes:
        image: The composited, canonically-banded ``ee.Image``.
        scene_count: Number of scenes contributing to the composite.
        source_datasets: Earth Engine collection ids used.
        year: Study year requested.
    """

    image: Any
    scene_count: int
    source_datasets: list[str]
    year: int


class BaseLoader(ABC):
    """Common workflow shared by all acquisition loaders.

    Subclasses implement :meth:`_build_collection` and
    :meth:`_harmonize_bands`; this base class handles AOI validation,
    cloud filtering thresholds, seasonal windowing, compositing, and
    scene-count validation.
    """

    #: Minimum acceptable cloud-free scenes for a valid composite.
    min_scenes: int = MIN_SCENES_REQUIRED
    #: Maximum acceptable cloud cover percentage at the filter stage.
    max_cloud_cover: float = MAX_CLOUD_COVER_PERCENT

    def __init__(self, settings: Settings, ee_manager: EarthEngineManager) -> None:
        self.settings = settings
        self.ee_manager = ee_manager

    @property
    @abstractmethod
    def label(self) -> str:
        """Short human-readable name for logging, e.g. ``'Landsat'``."""

    @abstractmethod
    def _build_collection(self, year: int, aoi: Any) -> tuple[Any, list[str]]:
        """Build the filtered ``ee.ImageCollection`` for a given year.

        Args:
            year: Study year.
            aoi: ``ee.Geometry`` AOI.

        Returns:
            Tuple of ``(collection, source_dataset_ids)``.
        """

    @abstractmethod
    def _harmonize_bands(self, collection: Any) -> Any:
        """Rename sensor-native bands to canonical names on every image.

        Args:
            collection: Raw ``ee.ImageCollection``.

        Returns:
            The collection with renamed/scaled bands.
        """

    def load(self, year: int, aoi: Any | None = None) -> LoaderResult:
        """Load, filter, harmonize, and composite imagery for a year.

        Args:
            year: Study year, must be one of ``config.constants.STUDY_YEARS``.
            aoi: Optional ``ee.Geometry``; defaults to the configured
                city AOI.

        Returns:
            A populated :class:`LoaderResult`.
        """
        self.ee_manager.ensure_initialized()
        aoi = aoi if aoi is not None else self.ee_manager.get_aoi_geometry()
        validate_aoi(aoi, f"{self.label} {year}")

        raw_collection, source_ids = self._build_collection(year, aoi)
        harmonized = self._harmonize_bands(raw_collection)

        scene_count = validate_collection_size(
            harmonized, self.min_scenes, f"{self.label} {year}"
        )

        composite = harmonized.median().clip(aoi)
        composite = composite.set(
            {
                "year": year,
                "sensor": self.label,
                "scene_count": scene_count,
                "source_datasets": source_ids,
            }
        )

        logger.info(
            "%s %d: composited %d scene(s) from %s.",
            self.label, year, scene_count, source_ids,
        )

        return LoaderResult(
            image=composite,
            scene_count=scene_count,
            source_datasets=source_ids,
            year=year,
        )

    def _date_range(self, year: int) -> tuple[str, str]:
        from config.constants import SEASON_END_MONTH_DAY, SEASON_START_MONTH_DAY

        return season_date_range(year, SEASON_START_MONTH_DAY, SEASON_END_MONTH_DAY)
