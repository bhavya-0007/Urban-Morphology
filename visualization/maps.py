"""Interactive HTML map generation using geemap/folium."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


class InteractiveMapBuilder:
    """Builds interactive Leaflet-based HTML maps of morphology layers."""

    def __init__(self, center_lat: float = 17.385, center_lon: float = 78.4867, zoom: int = 11) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom

    def build_ee_layer_map(self, layers: dict[str, tuple[Any, dict[str, Any]]]) -> Any:
        """Build a geemap Map with multiple Earth Engine layers.

        Args:
            layers: Mapping of layer name -> ``(ee.Image, vis_params)``.

        Returns:
            A ``geemap.Map`` instance with all layers added and a layer
            control enabled.
        """
        import geemap

        m = geemap.Map(center=[self.center_lat, self.center_lon], zoom=self.zoom)
        for name, (image, vis_params) in layers.items():
            m.addLayer(image, vis_params, name)
        m.addLayerControl()
        return m

    def save_html(self, map_obj: Any, output_path: Path) -> Path:
        """Save a geemap/folium map to a standalone HTML file.

        Args:
            map_obj: A ``geemap.Map`` or ``folium.Map``.
            output_path: Destination ``.html`` path.

        Returns:
            The path written to.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        map_obj.to_html(str(output_path)) if hasattr(map_obj, "to_html") else map_obj.save(str(output_path))
        logger.info("Saved interactive map: %s", output_path)
        return output_path

    def build_vector_map(self, gdf: Any, value_column: str, output_path: Path, title: str = "") -> Path:
        """Build a static choropleth map of a fishnet/vector layer.

        Args:
            gdf: A ``geopandas.GeoDataFrame`` with ``value_column``.
            value_column: Column to color cells by.
            output_path: Destination ``.png`` path.
            title: Optional map title.

        Returns:
            The path written to.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column=value_column, cmap="viridis", legend=True, ax=ax)
        ax.set_title(title or value_column)
        ax.set_axis_off()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved static vector map: %s", output_path)
        return output_path
