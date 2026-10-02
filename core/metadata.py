"""Metadata generation for every pipeline output.

Every exported artifact (raster, vector, model, figure) is accompanied by
a sidecar ``*.metadata.json`` describing provenance: which module
produced it, input parameters, timestamp, and software versions.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OutputMetadata:
    """Provenance record for a single exported artifact.

    Attributes:
        product: Short product name, e.g. ``"ndvi"``, ``"lcz_map"``.
        module: Fully-qualified module that generated the artifact.
        year: Study year the artifact corresponds to (if applicable).
        city: City key the artifact corresponds to.
        crs: Coordinate reference system of the artifact.
        parameters: Arbitrary key/value parameters used to produce it.
        source_datasets: Earth Engine / external dataset IDs consumed.
        created_at: ISO-8601 UTC timestamp.
        software: Python + key library versions for reproducibility.
    """

    product: str
    module: str
    year: int | None = None
    city: str | None = None
    crs: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    source_datasets: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    software: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)


def collect_software_versions() -> dict[str, str]:
    """Best-effort collection of installed package versions.

    Returns:
        Mapping of package name to version string. Packages that are not
        installed are simply omitted rather than raising.
    """
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("ee", "geemap", "geopandas", "rasterio", "xgboost", "shap", "sklearn"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return versions


def write_metadata(output_path: Path, metadata: OutputMetadata) -> Path:
    """Write a metadata sidecar file next to an output artifact.

    Args:
        output_path: Path to the artifact (e.g. ``ndvi_2020.tif``).
        metadata: The metadata record to serialize.

    Returns:
        Path to the written ``*.metadata.json`` file.
    """
    if not metadata.software:
        metadata.software = collect_software_versions()

    meta_path = Path(str(output_path) + ".metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, default=str)
    logger.debug("Wrote metadata: %s", meta_path)
    return meta_path
