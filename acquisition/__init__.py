"""Data acquisition loaders for all remote-sensing and ancillary sources."""
from __future__ import annotations

from acquisition.base import BaseLoader, LoaderResult
from acquisition.ghsl_loader import GHSLLoader
from acquisition.glcfcs_loader import GLCFCSLoader
from acquisition.landsat_loader import LandsatLoader
from acquisition.sentinel1_loader import Sentinel1Loader
from acquisition.sentinel2_loader import Sentinel2Loader

__all__ = [
    "BaseLoader",
    "LoaderResult",
    "LandsatLoader",
    "Sentinel2Loader",
    "Sentinel1Loader",
    "GHSLLoader",
    "GLCFCSLoader",
]
