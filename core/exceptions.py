"""Custom exception hierarchy for the urban morphology pipeline.

Using specific exception types (rather than bare ``Exception``) lets the
CLI and orchestration layer give actionable error messages and lets
callers catch precisely what they expect.
"""
from __future__ import annotations


class UrbanMorphologyError(Exception):
    """Base class for all project-specific errors."""


class EarthEngineAuthError(UrbanMorphologyError):
    """Raised when Earth Engine initialization/authentication fails."""


class EmptyCollectionError(UrbanMorphologyError):
    """Raised when an image collection has zero scenes after filtering."""


class InsufficientScenesError(UrbanMorphologyError):
    """Raised when fewer than the minimum required scenes are available."""


class MissingBandError(UrbanMorphologyError):
    """Raised when an expected band is absent from an image."""


class InvalidCRSError(UrbanMorphologyError):
    """Raised when a CRS is missing, malformed, or mismatched."""


class MissingAOIError(UrbanMorphologyError):
    """Raised when no area-of-interest geometry can be resolved."""


class ExportFailedError(UrbanMorphologyError):
    """Raised when an Earth Engine export task fails or times out."""


class ValidationError(UrbanMorphologyError):
    """Raised when a data/output validation check fails."""


class ConfigurationError(UrbanMorphologyError):
    """Raised for invalid or missing configuration values."""


class ModelTrainingError(UrbanMorphologyError):
    """Raised when ML training fails or produces an unusable model."""
