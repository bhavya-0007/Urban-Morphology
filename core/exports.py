"""Earth Engine export management.

Wraps ``ee.batch.Export`` calls (Drive / Asset) and local ``geemap``
downloads behind a single interface driven by ``Settings.export_target``,
with polling and structured failure reporting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import Settings
from core.exceptions import ExportFailedError
from core.logger import get_logger
from core.utils import safe_filename

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 15
_MAX_POLL_SECONDS = 60 * 60 * 6  # 6 hours ceiling for a single task


@dataclass
class ExportManager:
    """Coordinates exporting ``ee.Image``/``ee.FeatureCollection`` outputs.

    Attributes:
        settings: Resolved :class:`Settings` instance controlling the
            export destination.
    """

    settings: Settings

    def export_image(
        self,
        image: Any,
        description: str,
        region: Any,
        scale: int,
        crs: str,
        folder_parts: tuple[str, ...] = (),
        wait: bool = False,
    ) -> Path | str:
        """Export an ``ee.Image`` to the configured destination.

        Args:
            image: The ``ee.Image`` to export.
            description: Human-readable task/file name (sanitized
                internally).
            region: ``ee.Geometry`` export region.
            scale: Output pixel resolution in meters.
            crs: Target CRS, e.g. ``"EPSG:32644"``.
            folder_parts: Sub-folder path components under the export
                root (Drive folder or local ``outputs/``).
            wait: If ``True``, block and poll until the task completes
                (only meaningful for Drive/Asset targets).

        Returns:
            For ``export_target == "local"``, the local file path written.
            Otherwise, the Earth Engine task id string.

        Raises:
            ExportFailedError: If task creation or completion fails.
        """
        import ee

        name = safe_filename(description)

        if self.settings.export_target == "local":
            return self._export_local_image(image, name, region, scale, crs, folder_parts)

        if self.settings.export_target == "asset":
            asset_root = self.settings.ee_asset_root
            if not asset_root:
                raise ExportFailedError("EE_ASSET_ROOT must be set for asset exports.")
            asset_id = f"{asset_root}/{'/'.join(folder_parts)}/{name}".replace("//", "/")
            task = ee.batch.Export.image.toAsset(
                image=image,
                description=name[:100],
                assetId=asset_id,
                region=region,
                scale=scale,
                crs=crs,
                maxPixels=1e13,
            )
        else:  # drive
            drive_folder = "/".join((self.settings.drive_folder, *folder_parts))
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=name[:100],
                folder=drive_folder,
                fileNamePrefix=name,
                region=region,
                scale=scale,
                crs=crs,
                maxPixels=1e13,
                fileFormat="GeoTIFF",
            )

        task.start()
        logger.info("Started export task '%s' (id=%s).", name, task.id)

        if wait:
            self._wait_for_task(task, name)

        return task.id

    def export_table(
        self,
        feature_collection: Any,
        description: str,
        folder_parts: tuple[str, ...] = (),
        file_format: str = "GeoJSON",
        wait: bool = False,
    ) -> str:
        """Export an ``ee.FeatureCollection`` to the configured destination.

        Args:
            feature_collection: The ``ee.FeatureCollection`` to export.
            description: Human-readable task/file name.
            folder_parts: Sub-folder path components.
            file_format: One of ``"GeoJSON"``, ``"CSV"``, ``"SHP"``,
                ``"KML"``.
            wait: If ``True``, block until the task completes.

        Returns:
            The Earth Engine task id string.

        Raises:
            ExportFailedError: If task creation fails.
        """
        import ee

        name = safe_filename(description)
        drive_folder = "/".join((self.settings.drive_folder, *folder_parts))
        task = ee.batch.Export.table.toDrive(
            collection=feature_collection,
            description=name[:100],
            folder=drive_folder,
            fileNamePrefix=name,
            fileFormat=file_format,
        )
        task.start()
        logger.info("Started table export task '%s' (id=%s).", name, task.id)
        if wait:
            self._wait_for_task(task, name)
        return task.id

    def _export_local_image(
        self,
        image: Any,
        name: str,
        region: Any,
        scale: int,
        crs: str,
        folder_parts: tuple[str, ...],
    ) -> Path:
        """Download an ``ee.Image`` directly to local disk via ``geemap``."""
        try:
            import geemap
        except ImportError as exc:
            raise ExportFailedError(
                "geemap is required for local exports: pip install geemap"
            ) from exc

        out_dir = self.settings.output_dir(*folder_parts)
        out_path = out_dir / f"{name}.tif"
        try:
            geemap.ee_export_image(
                image, filename=str(out_path), scale=scale, region=region, crs=crs, file_per_band=False
            )
        except Exception as exc:  # noqa: BLE001
            raise ExportFailedError(f"Local export of '{name}' failed: {exc}") from exc
        logger.info("Exported locally: %s", out_path)
        return out_path

    def _wait_for_task(self, task: Any, name: str) -> None:
        """Poll an Earth Engine task until completion, failure, or timeout."""
        elapsed = 0
        while elapsed < _MAX_POLL_SECONDS:
            status = task.status()
            state = status.get("state")
            if state == "COMPLETED":
                logger.info("Task '%s' completed.", name)
                return
            if state in {"FAILED", "CANCELLED"}:
                raise ExportFailedError(
                    f"Task '{name}' ended with state={state}: {status.get('error_message')}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
            elapsed += _POLL_INTERVAL_SECONDS
        raise ExportFailedError(f"Task '{name}' timed out after {_MAX_POLL_SECONDS}s.")
