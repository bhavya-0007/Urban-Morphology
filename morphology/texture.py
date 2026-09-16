"""GLCM texture analysis.

Two implementations are provided: a server-side Earth Engine version
using ``ee.Image.glcmTexture`` (fast, coarse control), and a local
``scikit-image`` version for finer control over offsets/angles on
already-exported rasters.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from config.constants import GLCM_FEATURES
from core.logger import get_logger

logger = get_logger(__name__)

# Mapping from our canonical property names to the suffixes appended by
# ee.Image.glcmTexture to each input band (e.g. "nir_contrast").
_EE_GLCM_SUFFIX = {
    "contrast": "contrast",
    "entropy": "ent",
    "homogeneity": "idm",
    "correlation": "corr",
    "asm": "asm",
    "energy": "asm",  # GLCM "asm" (angular second moment) == energy^2 in EE's implementation
}


class GLCMTextureAnalyzer:
    """Computes GLCM texture features on a single grayscale-like band."""

    def __init__(self, size: int = 3, properties: list[str] | None = None) -> None:
        """
        Args:
            size: GLCM window radius (in pixels); EE window is
                ``(2*size+1) x (2*size+1)``.
            properties: Subset of texture properties to compute (case
                insensitive); defaults to
                :data:`config.constants.GLCM_FEATURES`.
        """
        self.size = size
        self.properties = [p.lower() for p in (properties or GLCM_FEATURES)]

    def compute_ee(self, image: Any, band: str) -> Any:
        """Compute GLCM texture bands server-side via Earth Engine.

        Args:
            image: An ``ee.Image``.
            band: Name of the band to compute texture on (e.g.
                ``"NDBI"`` or a grayscale luminance band).

        Returns:
            An ``ee.Image`` with one band per requested texture property,
            named ``f"{band}_{property}"``.
        """
        # GLCM requires an integer (typically 8-bit) input band.
        scaled = image.select(band).multiply(100).add(100).toInt().rename(band)
        glcm = scaled.glcmTexture(size=self.size)

        output_bands = []
        for prop in self.properties:
            suffix = _EE_GLCM_SUFFIX[prop]
            source_name = f"{band}_{suffix}"
            output_bands.append(glcm.select(source_name).rename(f"{band}_{prop}"))

        result = output_bands[0]
        for b in output_bands[1:]:
            result = result.addBands(b)
        return result

    def compute_local(
        self, array: np.ndarray, distances: list[int] | None = None, angles: list[float] | None = None
    ) -> dict[str, float]:
        """Compute GLCM texture statistics on a local numpy array (raster tile).

        Args:
            array: 2D uint8 numpy array (e.g. a rasterio-read window).
            distances: Pixel pair distances for the GLCM (default ``[1]``).
            angles: Pixel pair angles in radians (default 4 directions).

        Returns:
            Mapping of property name to its mean value across the
            requested distances/angles.
        """
        from skimage.feature import graycomatrix, graycoprops

        distances = distances or [1]
        angles = angles or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(
            array, distances=distances, angles=angles, levels=256, symmetric=True, normed=True
        )

        results: dict[str, float] = {}
        skimage_prop_map = {
            "contrast": "contrast",
            "homogeneity": "homogeneity",
            "correlation": "correlation",
            "energy": "energy",
            "asm": "ASM",
        }
        for prop, skimage_name in skimage_prop_map.items():
            if prop in self.properties:
                results[prop] = float(np.mean(graycoprops(glcm, skimage_name)))

        if "entropy" in self.properties:
            # scikit-image has no direct GLCM entropy; compute manually.
            glcm_norm = glcm / (glcm.sum(axis=(0, 1), keepdims=True) + 1e-12)
            entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-12), axis=(0, 1))
            results["entropy"] = float(np.mean(entropy))

        return results
