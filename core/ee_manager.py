"""Google Earth Engine session management.

Wraps ``ee.Initialize`` so the rest of the codebase never touches
authentication directly. Supports three auth modes: interactive user
credentials, a service account key file, or an already-authenticated
environment (e.g. Colab).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import Settings
from core.exceptions import EarthEngineAuthError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EarthEngineManager:
    """Manages a single Earth Engine session for the pipeline run.

    Attributes:
        settings: Resolved :class:`Settings` instance.
        initialized: Whether ``ee.Initialize`` has succeeded in this
            process.
    """

    settings: Settings
    initialized: bool = False

    def initialize(self) -> None:
        """Authenticate and initialize the Earth Engine API.

        Uses a service account if ``ee_service_account`` and
        ``ee_credentials_path`` are configured; otherwise falls back to
        interactive/browser-based user authentication.

        Raises:
            EarthEngineAuthError: If the ``earthengine-api`` package is
                missing or initialization fails for any reason.
        """
        if self.initialized:
            return

        try:
            import ee
        except ImportError as exc:  # pragma: no cover - environment issue
            raise EarthEngineAuthError(
                "The 'earthengine-api' package is not installed. "
                "Install it with `pip install earthengine-api`."
            ) from exc

        try:
            if self.settings.ee_service_account and self.settings.ee_credentials_path:
                logger.info("Authenticating Earth Engine via service account.")
                credentials = ee.ServiceAccountCredentials(
                    self.settings.ee_service_account,
                    self.settings.ee_credentials_path,
                )
                ee.Initialize(credentials, project=self.settings.ee_project)
            else:
                logger.info("Authenticating Earth Engine via user credentials.")
                try:
                    ee.Initialize(project=self.settings.ee_project)
                except Exception:
                    logger.info("No cached credentials found — launching ee.Authenticate().")
                    ee.Authenticate()
                    ee.Initialize(project=self.settings.ee_project)
        except Exception as exc:  # noqa: BLE001 - surfaced as domain error
            raise EarthEngineAuthError(f"Earth Engine initialization failed: {exc}") from exc

        self.initialized = True
        logger.info("Earth Engine initialized for project '%s'.", self.settings.ee_project)

    def ensure_initialized(self) -> None:
        """Initialize on first use; no-op afterward."""
        if not self.initialized:
            self.initialize()

    def get_aoi_geometry(self) -> Any:
        """Build an ``ee.Geometry`` rectangle for the configured city AOI.

        Returns:
            An ``ee.Geometry.Rectangle`` built from the city's WGS84
            bounding box.
        """
        self.ensure_initialized()
        import ee

        bbox = self.settings.city.bbox_wgs84
        return ee.Geometry.Rectangle(list(bbox))


_MANAGER: EarthEngineManager | None = None


def get_ee_manager(settings: Settings) -> EarthEngineManager:
    """Return a process-wide singleton :class:`EarthEngineManager`.

    Args:
        settings: Settings used to construct the manager if not already
            created.

    Returns:
        The shared :class:`EarthEngineManager` instance.
    """
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = EarthEngineManager(settings=settings)
    return _MANAGER
