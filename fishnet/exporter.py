"""Export the final fishnet feature dataset to GeoPackage/CSV/GeoJSON."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.logger import get_logger
from core.metadata import OutputMetadata, write_metadata
from core.validation import validate_dataframe_not_empty

logger = get_logger(__name__)


class FishnetExporter:
    """Writes the aggregated fishnet feature table to disk in multiple formats."""

    def export(
        self,
        fishnet_gdf: Any,
        output_dir: Path,
        base_name: str,
        year: int,
        city: str,
        crs: str,
        formats: tuple[str, ...] = ("gpkg", "csv", "geojson", "shp"),
    ) -> dict[str, Path]:
        """Write the fishnet dataset to the requested formats.

        Args:
            fishnet_gdf: The merged fishnet ``geopandas.GeoDataFrame``.
            output_dir: Destination directory (created if needed).
            base_name: Base filename, e.g. ``"fishnet_features_2020"``.
            year: Study year, recorded in metadata.
            city: City key, recorded in metadata.
            crs: CRS string, recorded in metadata.
            formats: Which formats to write: any of
                ``"gpkg"``, ``"csv"``, ``"geojson"``.

        Returns:
            Mapping of format -> written file path.
        """
        validate_dataframe_not_empty(fishnet_gdf, f"fishnet export {base_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}

        if "gpkg" in formats:
            path = output_dir / f"{base_name}.gpkg"
            fishnet_gdf.to_file(path, driver="GPKG")
            written["gpkg"] = path

        if "geojson" in formats:
            path = output_dir / f"{base_name}.geojson"
            fishnet_gdf.to_file(path, driver="GeoJSON")
            written["geojson"] = path

        if "csv" in formats:
            path = output_dir / f"{base_name}.csv"
            non_geom_cols = [c for c in fishnet_gdf.columns if c != "geometry"]
            fishnet_gdf[non_geom_cols].to_csv(path, index=False)
            written["csv"] = path

        if "shp" in formats:
            path = output_dir / f"{base_name}.shp"
            fishnet_gdf.to_file(path, driver="ESRI Shapefile")
            written["shp"] = path

        for fmt, path in written.items():
            metadata = OutputMetadata(
                product=f"fishnet_features_{fmt}",
                module="fishnet.exporter.FishnetExporter",
                year=year,
                city=city,
                crs=crs,
                parameters={"n_cells": int(len(fishnet_gdf)), "columns": list(fishnet_gdf.columns)},
            )
            write_metadata(path, metadata)

        logger.info("Exported fishnet dataset '%s' in formats: %s", base_name, list(written))
        return written
