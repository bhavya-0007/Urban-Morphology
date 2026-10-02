"""Project-wide constants that do not change per-run.

These are physical/scientific constants and fixed enumerations used
throughout the pipeline. Anything that is environment- or run-specific
belongs in settings.py instead.
"""
from __future__ import annotations

from enum import Enum


class Season(str, Enum):
    """Supported compositing seasons."""

    PRE_MONSOON = "pre_monsoon"


# Pre-monsoon window used for all seasonal composites (month-day).
SEASON_WINDOWS = {
    Season.PRE_MONSOON: {"start_month": 2, "start_day": 1, "end_month": 5, "end_day": 31},
}

# Cloud cover ceiling (percent) applied at the collection-filter stage.
MAX_CLOUD_COVER_PERCENT = 30

# Minimum number of cloud-filtered scenes required to build a valid
# seasonal composite.
MIN_SCENES_REQUIRED = 2

# Convenience month-day strings for the configured pre-monsoon window,
# derived from SEASON_WINDOWS so there is a single source of truth.
_PRE_MONSOON = SEASON_WINDOWS[Season.PRE_MONSOON]
SEASON_START_MONTH_DAY = f"{_PRE_MONSOON['start_month']:02d}-{_PRE_MONSOON['start_day']:02d}"
SEASON_END_MONTH_DAY = f"{_PRE_MONSOON['end_month']:02d}-{_PRE_MONSOON['end_day']:02d}"

# Standard nodata / fill value used across exported rasters.
NODATA_VALUE = -9999

# Fishnet grid cell size (meters), matching the PDD's 300 m specification.
FISHNET_CELL_SIZE = 300

# Native resolutions (meters) for each sensor family.
LANDSAT_RESOLUTION = 30
SENTINEL2_RESOLUTION = 10
SENTINEL1_RESOLUTION = 10

# Random seed used across sampling, train/test splits, and models.
RANDOM_SEED = 42

# LCZ class codes used throughout morphology + ML modules.
class LCZClass(str, Enum):
    LCZ_1 = "LCZ_1"    # Compact high-rise
    LCZ_2 = "LCZ_2"    # Compact mid-rise
    LCZ_3 = "LCZ_3"    # Compact low-rise
    LCZ_4 = "LCZ_4"    # Open high-rise
    LCZ_5 = "LCZ_5"    # Open mid-rise
    LCZ_6 = "LCZ_6"    # Open low-rise
    LCZ_8 = "LCZ_8"    # Large low-rise
    LCZ_10 = "LCZ_10"  # Heavy industry
    LCZ_A = "LCZ_A"    # Dense trees
    LCZ_D = "LCZ_D"    # Low plants


LCZ_CLASSES = [c.value for c in LCZClass]

# Spectral index registry (name -> required bands, generic across sensors).
SPECTRAL_INDICES = ["NDVI", "NDBI", "MNDWI", "IBI", "BSI", "UI", "ISF"]

# GLCM texture features computed on the morphology layer.
GLCM_FEATURES = ["contrast", "entropy", "homogeneity", "correlation", "energy"]

# Landscape metrics computed via PyLandStats.
LANDSCAPE_METRICS = [
    "patch_density",
    "edge_density",
    "mean_patch_size",
    "aggregation_index",
    "contagion",
    "shannon_diversity",
    "fractal_dimension",
    "landscape_shape_index",
]

EARTH_ENGINE_SCOPES = ["https://www.googleapis.com/auth/earthengine"]
