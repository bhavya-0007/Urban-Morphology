"""City-level configuration.

Only Hyderabad is configured per the PDD, but the structure supports
adding further cities without touching downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CityConfig:
    """Static configuration describing a study city."""

    name: str
    state: str
    country: str
    epsg: int
    # Bounding box in WGS84 (minx, miny, maxx, maxy). Used only as a
    # fallback if no AOI asset/shapefile is supplied at runtime.
    bbox_wgs84: tuple[float, float, float, float]
    fishnet_cell_size_m: int = 300


CITIES: dict[str, CityConfig] = {
    "hyderabad": CityConfig(
        name="Hyderabad",
        state="Telangana",
        country="India",
        epsg=32644,
        bbox_wgs84=(78.20, 17.20, 78.70, 17.60),
        fishnet_cell_size_m=300,
    ),
}


def get_city(city_key: str) -> CityConfig:
    """Fetch a city configuration by key.

    Args:
        city_key: Lowercase city identifier, e.g. ``"hyderabad"``.

    Returns:
        The matching :class:`CityConfig`.

    Raises:
        KeyError: If the city is not registered.
    """
    try:
        return CITIES[city_key.lower()]
    except KeyError as exc:
        raise KeyError(
            f"City '{city_key}' is not configured. Available: {list(CITIES)}"
        ) from exc
