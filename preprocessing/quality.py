"""Composite-level quality validation checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger
from core.validation import validate_value_range

logger = get_logger(__name__)


@dataclass
class QualityReport:
    """Summary of QA checks performed on a composite.

    Attributes:
        valid_pixel_fraction: Fraction of AOI pixels with valid data.
        band_ranges: Per-band ``(min, max)`` sampled statistics.
        passed: Whether all checks passed.
        issues: Human-readable descriptions of any failed checks.
    """

    valid_pixel_fraction: float
    band_ranges: dict[str, tuple[float, float]]
    passed: bool
    issues: list[str]


def run_quality_checks(
    image: Any,
    region: Any,
    scale: int,
    min_valid_fraction: float = 0.6,
    expected_ranges: dict[str, tuple[float, float]] | None = None,
) -> QualityReport:
    """Run standard QA checks on a composited image before export.

    Args:
        image: The composited ``ee.Image``.
        region: ``ee.Geometry`` region for statistics.
        scale: Pixel scale in meters for reductions.
        min_valid_fraction: Minimum acceptable fraction of valid pixels.
        expected_ranges: Optional mapping of band name to expected
            ``(min, max)`` for sanity checking (e.g. reflectance in
            ``[0, 1]``).

    Returns:
        A populated :class:`QualityReport`.
    """
    import ee

    issues: list[str] = []

    band_names = image.bandNames().getInfo()
    first_band = band_names[0]
    mask_stats = image.select([first_band]).mask().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e13
    ).getInfo()
    valid_fraction = float(mask_stats.get(first_band, 0.0) or 0.0)

    if valid_fraction < min_valid_fraction:
        issues.append(
            f"Valid pixel fraction {valid_fraction:.2f} below threshold {min_valid_fraction}."
        )

    band_ranges: dict[str, tuple[float, float]] = {}
    minmax = image.reduceRegion(
        reducer=ee.Reducer.minMax(), geometry=region, scale=scale, maxPixels=1e13, bestEffort=True
    ).getInfo()
    for band in band_names:
        lo = minmax.get(f"{band}_min")
        hi = minmax.get(f"{band}_max")
        if lo is None or hi is None:
            continue
        band_ranges[band] = (float(lo), float(hi))

        if expected_ranges and band in expected_ranges:
            exp_lo, exp_hi = expected_ranges[band]
            try:
                validate_value_range(lo, exp_lo - 0.05, exp_hi + 0.05, f"band '{band}' min")
                validate_value_range(hi, exp_lo - 0.05, exp_hi + 0.05, f"band '{band}' max")
            except Exception as exc:  # noqa: BLE001
                issues.append(str(exc))

    report = QualityReport(
        valid_pixel_fraction=valid_fraction,
        band_ranges=band_ranges,
        passed=len(issues) == 0,
        issues=issues,
    )
    if not report.passed:
        logger.warning("Quality checks failed: %s", issues)
    else:
        logger.info("Quality checks passed (valid_fraction=%.2f).", valid_fraction)
    return report
