"""Earth Engine dataset registry.

Central lookup of collection IDs and band mappings so that acquisition
loaders never hardcode strings inline.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    """Describes a single Earth Engine collection and its usable bands."""

    collection_id: str
    resolution_m: int
    bands: dict[str, str] = field(default_factory=dict)  # canonical -> native band name
    scale_factor: float = 1.0
    offset: float = 0.0
    cloud_band: str | None = None


LANDSAT_COLLECTIONS: dict[str, DatasetSpec] = {
    "LT05": DatasetSpec(
        collection_id="LANDSAT/LT05/C02/T1_L2",
        resolution_m=30,
        bands={"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
               "nir": "SR_B4", "swir1": "SR_B5", "swir2": "SR_B7"},
        scale_factor=0.0000275,
        offset=-0.2,
        cloud_band="QA_PIXEL",
    ),
    "LE07": DatasetSpec(
        collection_id="LANDSAT/LE07/C02/T1_L2",
        resolution_m=30,
        bands={"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
               "nir": "SR_B4", "swir1": "SR_B5", "swir2": "SR_B7"},
        scale_factor=0.0000275,
        offset=-0.2,
        cloud_band="QA_PIXEL",
    ),
    "LC08": DatasetSpec(
        collection_id="LANDSAT/LC08/C02/T1_L2",
        resolution_m=30,
        bands={"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
               "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7"},
        scale_factor=0.0000275,
        offset=-0.2,
        cloud_band="QA_PIXEL",
    ),
    "LC09": DatasetSpec(
        collection_id="LANDSAT/LC09/C02/T1_L2",
        resolution_m=30,
        bands={"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
               "nir": "SR_B5", "swir1": "SR_B6", "swir2": "SR_B7"},
        scale_factor=0.0000275,
        offset=-0.2,
        cloud_band="QA_PIXEL",
    ),
}

# Year -> ordered list of Landsat collections valid for that acquisition year.
# Ordering matters: later entries are preferred when multiple sensors overlap.
LANDSAT_YEAR_MAP: dict[int, list[str]] = {
    2000: ["LT05", "LE07"],
    2005: ["LT05", "LE07"],
    2010: ["LT05", "LE07"],
    2015: ["LC08"],
    2020: ["LC08"],
    2025: ["LC08", "LC09"],
}

SENTINEL2_SPEC = DatasetSpec(
    collection_id="COPERNICUS/S2_SR_HARMONIZED",
    resolution_m=10,
    bands={"blue": "B2", "green": "B3", "red": "B4", "nir": "B8",
           "swir1": "B11", "swir2": "B12"},
    scale_factor=0.0001,
    offset=0.0,
    cloud_band="QA60",
)

SENTINEL1_SPEC = DatasetSpec(
    collection_id="COPERNICUS/S1_GRD",
    resolution_m=10,
    bands={"vv": "VV", "vh": "VH"},
)

GHSL_BUILT_SPEC = DatasetSpec(
    collection_id="JRC/GHSL/P2023A/GHS_BUILT_S",
    resolution_m=100,   # was 10 — confirmed via the 10,000 m² max
)

GLC_FCS30D_SPEC = DatasetSpec(
    collection_id="projects/sat-io/open-datasets/GLC-FCS30D/annual",
    resolution_m=30,
)

STUDY_YEARS: list[int] = [2000, 2005, 2010, 2015, 2020, 2025]
