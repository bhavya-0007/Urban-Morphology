"""Building spacing metrics using OSM footprints and/or street networks.

Nearest-neighbor spacing is computed client-side with GeoPandas/Shapely
after pulling OSM building footprints for the AOI, since this kind of
geometric nearest-neighbor analysis is far more natural outside Earth
Engine's server-side paradigm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.exceptions import ValidationError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpacingStats:
    """Summary statistics of building-to-building spacing.

    Attributes:
        mean_spacing_m: Mean nearest-neighbor centroid distance (meters).
        median_spacing_m: Median nearest-neighbor centroid distance.
        std_spacing_m: Standard deviation of nearest-neighbor distances.
        building_count: Number of buildings included in the statistics.
    """

    mean_spacing_m: float
    median_spacing_m: float
    std_spacing_m: float
    building_count: int


class BuildingSpacingCalculator:
    """Computes nearest-neighbor building spacing from footprint polygons."""

    def compute(self, buildings_gdf: Any, projected_crs: str) -> SpacingStats:
        """Compute spacing statistics from a GeoDataFrame of buildings.

        Args:
            buildings_gdf: A ``geopandas.GeoDataFrame`` of building
                polygons (any CRS; will be reprojected).
            projected_crs: Metric CRS to reproject into, e.g.
                ``"EPSG:32644"``.

        Returns:
            A populated :class:`SpacingStats`.

        Raises:
            ValidationError: If fewer than 2 buildings are present.
        """
        import numpy as np
        from scipy.spatial import cKDTree

        if len(buildings_gdf) < 2:
            raise ValidationError("Need at least 2 buildings to compute spacing.")

        projected = buildings_gdf.to_crs(projected_crs)
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in projected.geometry])

        tree = cKDTree(centroids)
        # k=2 because the nearest neighbor of a point to itself is itself.
        distances, _ = tree.query(centroids, k=2)
        nearest_distances = distances[:, 1]

        return SpacingStats(
            mean_spacing_m=float(np.mean(nearest_distances)),
            median_spacing_m=float(np.median(nearest_distances)),
            std_spacing_m=float(np.std(nearest_distances)),
            building_count=len(projected),
        )

    def compute_per_fishnet_cell(
        self, buildings_gdf: Any, fishnet_gdf: Any, projected_crs: str
    ) -> Any:
        """Compute mean building spacing aggregated to each fishnet cell.

        Args:
            buildings_gdf: GeoDataFrame of building polygons.
            fishnet_gdf: GeoDataFrame of fishnet grid cells.
            projected_crs: Metric CRS for distance computation.

        Returns:
            The ``fishnet_gdf`` with an added ``building_spacing_m``
            column (``NaN`` for cells with fewer than 2 buildings).
        """
        import geopandas as gpd
        import numpy as np
        from scipy.spatial import cKDTree

        buildings = buildings_gdf.to_crs(projected_crs)
        fishnet = fishnet_gdf.to_crs(projected_crs)

        buildings["centroid"] = buildings.geometry.centroid
        joined = gpd.sjoin(
            gpd.GeoDataFrame(buildings, geometry="centroid", crs=projected_crs),
            fishnet[["geometry"]].reset_index().rename(columns={"index": "cell_id"}),
            predicate="within",
        )

        spacing_by_cell: dict[int, float] = {}
        for cell_id, group in joined.groupby("cell_id"):
            if len(group) < 2:
                continue
            coords = np.array([[pt.x, pt.y] for pt in group["centroid"]])
            tree = cKDTree(coords)
            distances, _ = tree.query(coords, k=2)
            spacing_by_cell[cell_id] = float(np.mean(distances[:, 1]))

        fishnet_gdf = fishnet_gdf.copy()
        fishnet_gdf["building_spacing_m"] = fishnet_gdf.index.map(spacing_by_cell)
        return fishnet_gdf
