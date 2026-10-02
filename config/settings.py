"""Runtime settings for the urban morphology pipeline.

Values here are meant to be overridden via environment variables or a
``.env`` file for machine-specific paths (Earth Engine project id, output
directories), while algorithmic/scientific constants stay in
``constants.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config.cities import CityConfig, get_city
from config.constants import Season

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    """Aggregated, immutable run configuration.

    An instance of this class is threaded through every module instead of
    reading environment variables ad hoc, which keeps the pipeline
    testable and configuration-driven per the coding standards.
    """
    study_years: tuple[int, ...] = (
    2000,
    2005,
    2010,
    2015,
    2020,
    2025,
)

    # Google Earth Engine
    ee_project: str = field(default_factory=lambda: _env("EE_PROJECT", "orbital-eon-490107-m0"))
    ee_service_account: str | None = field(
        default_factory=lambda: os.environ.get("EE_SERVICE_ACCOUNT")
    )
    ee_credentials_path: str | None = field(
        default_factory=lambda: os.environ.get("EE_CREDENTIALS_PATH")
    )

    # City / AOI
    city_key: str = field(default_factory=lambda: _env("CITY_KEY", "hyderabad"))

    # Projection
    crs: str = field(default_factory=lambda: _env("PROJECT_CRS", "EPSG:32644"))

    # Season
    season: Season = Season.PRE_MONSOON

    # Output locations
    output_root: Path = field(
        default_factory=lambda: Path(_env("OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))
    )

    # Export destination: "drive" (Google Drive), "asset" (EE asset), or "local"
    export_target: str = field(default_factory=lambda: _env("EXPORT_TARGET", "drive"))
    drive_folder: str = field(default_factory=lambda: _env("DRIVE_FOLDER", "urban_morphology_hyderabad"))
    ee_asset_root: str | None = field(default_factory=lambda: os.environ.get("EE_ASSET_ROOT"))

    # Logging
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    log_dir: Path = field(
        default_factory=lambda: Path(_env("LOG_DIR", str(PROJECT_ROOT / "outputs" / "logs")))
    )

    # ML
    random_seed: int = 42
    test_size: float = 0.2

    @property
    def city(self) -> CityConfig:
        """Resolved :class:`CityConfig` for ``city_key``."""
        return get_city(self.city_key)

    def output_dir(self, *parts: str) -> Path:
        """Build (and create) a sub-directory under ``output_root``.

        Args:
            *parts: Path components appended to ``output_root``.

        Returns:
            The created directory path.
        """
        path = self.output_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_settings() -> Settings:
    """Factory returning a fresh :class:`Settings` snapshot.

    Kept as a function (rather than a module-level singleton) so tests can
    monkeypatch environment variables and get an independent instance.
    """
    return Settings()
