"""Aggregate raster/vector morphology layers onto fishnet grid cells.

Two paths are supported:
  1. Server-side via ``ee.Image.reduceRegions`` against an
     ``ee.FeatureCollection`` fishnet (fast, scales well, no local
     raster download needed).
  2. Local via ``rasterstats``/``rioxarray`` zonal statistics against a
     GeoDataFrame fishnet and locally-exported GeoTIFFs (used when
     layers were already exported, e.g. texture or landscape-metric
     rasters computed outside Earth Engine).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.exceptions import ValidationError
from core.logger import get_logger
from core.validation import validate_dataframe_not_empty

logger = get_logger(__name__)


class FishnetAggregator:
    """Aggregates raster layers (NDVI, NDBI, ISF, texture, LCZ, ...) onto
    fishnet cells, producing one row per cell with one column per layer.
    """

    def aggregate_ee(
        self,
        fishnet_fc: Any,
        image: Any,
        band_names: list[str] | None = None,
        reducer: Any | None = None,
        scale: int = 10,
    ) -> Any:
        """Aggregate an ``ee.Image`` onto an ``ee.FeatureCollection`` fishnet.

        Args:
            fishnet_fc: ``ee.FeatureCollection`` of fishnet cells.
            image: Multi-band ``ee.Image`` to summarize per cell.
            band_names: Optional subset of bands to reduce; defaults to
                all bands in ``image``.
            reducer: ``ee.Reducer`` to apply; defaults to ``mean()``.
            scale: Pixel scale in meters for the reduction.

        Returns:
            An ``ee.FeatureCollection`` with one property per band added
            to each fishnet feature.
        """
        import ee

        reducer = reducer or ee.Reducer.mean()
        img = image.select(band_names) if band_names else image
        return img.reduceRegions(collection=fishnet_fc, reducer=reducer, scale=scale)

    def aggregate_local(
        self,
        fishnet_gdf: Any,
        raster_path: Path,
        column_name: str,
        stat: str = "mean",
    ) -> Any:
        """Zonal-statistics aggregation of a local raster onto a fishnet.

        Args:
            fishnet_gdf: ``geopandas.GeoDataFrame`` of fishnet cells.
            raster_path: Path to a single-band GeoTIFF.
            column_name: Name of the output column to add.
            stat: One of ``rasterstats`` supported stats (``"mean"``,
                ``"median"``, ``"majority"``, ``"std"``, etc.).

        Returns:
            ``fishnet_gdf`` with ``column_name`` added.

        Raises:
            ValidationError: If ``rasterstats`` is not installed or the
                raster cannot be read.
        """
        try:
            from rasterstats import zonal_stats
        except ImportError as exc:
            raise ValidationError(
                "rasterstats is required for local aggregation: pip install rasterstats"
            ) from exc

        try:
            stats = zonal_stats(fishnet_gdf, str(raster_path), stats=[stat], nodata=None)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"Zonal stats failed for {raster_path}: {exc}") from exc

        fishnet_gdf = fishnet_gdf.copy()
        fishnet_gdf[column_name] = [row.get(stat) for row in  stats]
        return fishnet_gdf 

    def merge_layers(self, fishnet_gdf: Any, layer_frames: dict[str, Any]) -> Any:
        """Merge multiple per-layer aggregation results into one wide table.

        Args:
            fishnet_gdf: Base fishnet ``geopandas.GeoDataFrame`` with
                ``cell_id``.
            layer_frames: Mapping of layer name -> DataFrame containing
                ``cell_id`` and one or more value columns.

        Returns:
            A single merged ``geopandas.GeoDataFrame``.
        """
        merged = fishnet_gdf
        for name, frame in layer_frames.items():
            merged = merged.merge(frame, on="cell_id", how="left", suffixes=("", f"_{name}"))
        validate_dataframe_not_empty(merged, "fishnet aggregation merge")
        return merged
    
