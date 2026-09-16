"""Landscape metrics using PyLandStats.

Operates on locally-available categorical rasters (e.g. an exported
morphology-class GeoTIFF), since PyLandStats requires a numpy array /
local raster rather than an Earth Engine server-side image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.constants import LANDSCAPE_METRICS
from core.exceptions import ValidationError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LandscapeMetricsResult:
    """Landscape-level metric results.

    Attributes:
        landscape_metrics: Mapping of metric name to value at the
            landscape level.
        class_metrics: Per-class metric DataFrame (class_val -> metrics),
            kept as a dict-of-dicts for JSON-friendliness.
    """

    landscape_metrics: dict[str, float]
    class_metrics: dict[int, dict[str, float]]


class LandscapeMetricsCalculator:
    """Wraps PyLandStats to compute the metrics required by the PDD."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or LANDSCAPE_METRICS

    def compute_from_raster(self, raster_path: Path, nodata: int | None = 255) -> LandscapeMetricsResult:
        """Compute landscape metrics from a categorical class raster.

        Args:
            raster_path: Path to a single-band categorical GeoTIFF (e.g.
                morphology typology or LCZ classes).
            nodata: Nodata value in the raster to exclude from analysis.

        Returns:
            A populated :class:`LandscapeMetricsResult`.

        Raises:
            ValidationError: If the raster cannot be read or is empty.
        """
        try:
            import pylandstats as pls
        except ImportError as exc:
            raise ValidationError(
                "pylandstats is required for landscape metrics: pip install pylandstats"
            ) from exc

        try:
            landscape = pls.Landscape(str(raster_path), nodata=nodata)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"Failed to load raster for landscape metrics: {exc}") from exc

        metric_name_map = {
            "patch_density": "patch_density",
            "edge_density": "edge_density",
            "mean_patch_size": "area_mn",
            "aggregation_index": "ai",
            "contagion": "contag",
            "shannon_diversity": "shdi",
            "fractal_dimension": "frac_mn",
            "landscape_shape_index": "lsi",
        }

        landscape_level: dict[str, float] = {}
        for metric in self.metrics:
            pls_name = metric_name_map.get(metric)
            if pls_name is None:
                logger.warning("Unknown landscape metric '%s', skipping.", metric)
                continue
            try:
                method = getattr(landscape, pls_name)
                value = method()
                landscape_level[metric] = float(value)
            except AttributeError:
                logger.warning("PyLandStats has no landscape-level method '%s'.", pls_name)

        class_metrics_df = landscape.compute_class_metrics_df(
            metrics=["area_mn", "pd", "ed", "ai"]
        )
        class_metrics: dict[int, dict[str, float]] = {
            int(idx): row.to_dict() for idx, row in class_metrics_df.iterrows()
        }

        return LandscapeMetricsResult(
            landscape_metrics=landscape_level, class_metrics=class_metrics
        )
