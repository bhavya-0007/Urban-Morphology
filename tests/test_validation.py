"""Tests for core.validation helpers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.exceptions import (
    EmptyCollectionError,
    InsufficientScenesError,
    InvalidCRSError,
    MissingAOIError,
    MissingBandError,
    ValidationError,
)
from core.validation import (
    validate_aoi,
    validate_bands,
    validate_collection_size,
    validate_crs,
    validate_dataframe_not_empty,
    validate_value_range,
)


def test_validate_collection_size_empty_raises():
    collection = MagicMock()
    collection.size.return_value.getInfo.return_value = 0
    with pytest.raises(EmptyCollectionError):
        validate_collection_size(collection, min_scenes=1, label="test")


def test_validate_collection_size_insufficient_raises():
    collection = MagicMock()
    collection.size.return_value.getInfo.return_value = 1
    with pytest.raises(InsufficientScenesError):
        validate_collection_size(collection, min_scenes=3, label="test")


def test_validate_collection_size_passes():
    collection = MagicMock()
    collection.size.return_value.getInfo.return_value = 5
    assert validate_collection_size(collection, min_scenes=2, label="test") == 5


def test_validate_bands_missing_raises():
    image = MagicMock()
    image.bandNames.return_value.getInfo.return_value = ["red", "green"]
    with pytest.raises(MissingBandError):
        validate_bands(image, ["red", "nir"], label="test")


def test_validate_bands_passes():
    image = MagicMock()
    image.bandNames.return_value.getInfo.return_value = ["red", "nir", "green"]
    validate_bands(image, ["red", "nir"], label="test")  # should not raise


@pytest.mark.parametrize("crs", [None, "", "bogus", "UTM:32644"])
def test_validate_crs_invalid_raises(crs):
    with pytest.raises(InvalidCRSError):
        validate_crs(crs, label="test")


def test_validate_crs_valid_passes():
    assert validate_crs("EPSG:32644", label="test") == "EPSG:32644"


def test_validate_aoi_none_raises():
    with pytest.raises(MissingAOIError):
        validate_aoi(None, label="test")


def test_validate_aoi_zero_area_raises():
    geometry = MagicMock()
    geometry.area.return_value.getInfo.return_value = 0
    with pytest.raises(MissingAOIError):
        validate_aoi(geometry, label="test")


def test_validate_dataframe_not_empty_raises():
    with pytest.raises(ValidationError):
        validate_dataframe_not_empty([], label="test")


def test_validate_value_range_out_of_bounds_raises():
    with pytest.raises(ValidationError):
        validate_value_range(1.5, 0.0, 1.0, label="test")


def test_validate_value_range_within_bounds_passes():
    validate_value_range(0.5, 0.0, 1.0, label="test")  # should not raise
