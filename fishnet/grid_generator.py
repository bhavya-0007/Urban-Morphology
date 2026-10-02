"""Fishnet grid generation over the study AOI (default 300 m cells)."""
from __future__ import annotations

from typing import Any

from config.constants import FISHNET_CELL_SIZE
from core.logger import get_logger

logger = get_logger(__name__)


class FishnetGenerator:
    """Generates a regular square-cell fishnet grid clipped to an AOI."""

    def __init__(self, cell_size_m: int = FISHNET_CELL_SIZE) -> None:
        self.cell_size_m = cell_size_m

    def generate(self, aoi_gdf: Any, crs: str) -> Any:
        """Build a fishnet GeoDataFrame covering ``aoi_gdf``.

        Args:
            aoi_gdf: A ``geopandas.GeoDataFrame`` (any CRS) describing the
                study area boundary.
            crs: Metric CRS to build the grid in, e.g. ``"EPSG:32644"``.

        Returns:
            A ``geopandas.GeoDataFrame`` of square polygons intersecting
            the AOI, indexed sequentially, with a ``cell_id`` column.
        """
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import box

        projected = aoi_gdf.to_crs(crs)
        minx, miny, maxx, maxy = projected.total_bounds

        xs = np.arange(minx, maxx, self.cell_size_m)
        ys = np.arange(miny, maxy, self.cell_size_m)

        cells = [
            box(x, y, x + self.cell_size_m, y + self.cell_size_m)
            for x in xs
            for y in ys
        ]
        fishnet = gpd.GeoDataFrame({"geometry": cells}, crs=crs)

        union = projected.geometry.union_all() if hasattr(projected.geometry, "union_all") else projected.unary_union
        fishnet = fishnet[fishnet.intersects(union)].reset_index(drop=True)
        fishnet["cell_id"] = fishnet.index
        fishnet["cell_area_m2"] = fishnet.geometry.area

        logger.info(
            "Generated fishnet: %d cells at %dm resolution.", len(fishnet), self.cell_size_m
        )
        return fishnet

    def generate_ee(self, aoi: Any) -> Any:
        """Build the fishnet server-side as an ``ee.FeatureCollection``.

        Useful when aggregation will also happen server-side via
        ``reduceRegions`` rather than after a local export.

        Args:
            aoi: ``ee.Geometry`` AOI.

        Returns:
            An ``ee.FeatureCollection`` of square cell features.
        """
        import ee

        proj = ee.Projection("EPSG:32644").atScale(self.cell_size_m)
        grid = aoi.coveringGrid(proj, self.cell_size_m)

        grid = grid.toList(grid.size())

        def make_feature(i):
            feature = ee.Feature(grid.get(i))
            return feature.set("cell_id", i)

        return ee.FeatureCollection(
            ee.List.sequence(0, ee.Number(grid.size()).subtract(1))
            .map(make_feature)
        )
