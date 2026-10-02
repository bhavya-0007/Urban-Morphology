"""Spectral index calculators.

Each index is implemented as an independent class (see the PDD §12
requirement). ``ALL_INDICES`` provides a name -> class registry used by
the CLI's ``indices`` command to compute every index in one pass.
"""
from __future__ import annotations

from indices.bsi import BSI
from indices.ibi import IBI
from indices.isf import ISF
from indices.mndwi import MNDWI
from indices.ndbi import NDBI
from indices.ndvi import NDVI
from indices.ui import UI

ALL_INDICES: dict[str, type] = {
    "NDVI": NDVI,
    "NDBI": NDBI,
    "MNDWI": MNDWI,
    "IBI": IBI,
    "BSI": BSI,
    "UI": UI,
    "ISF": ISF,
}

__all__ = ["NDVI", "NDBI", "MNDWI", "IBI", "BSI", "UI", "ISF", "ALL_INDICES"]
