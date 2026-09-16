"""Tests for spectral index formula correctness (pure-math, EE mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from indices import ALL_INDICES
from indices.bsi import BSI
from indices.ibi import IBI
from indices.isf import ISF
from indices.mndwi import MNDWI
from indices.ndbi import NDBI
from indices.ndvi import NDVI


def _mock_image_with_bands(band_names: list[str]) -> MagicMock:
    """Build a MagicMock standing in for an ee.Image with given bands."""
    image = MagicMock()
    image.bandNames.return_value.getInfo.return_value = band_names
    image.select.return_value = image
    image.normalizedDifference.return_value = image
    image.expression.return_value = image
    image.rename.return_value = image
    image.copyProperties.return_value = image
    image.propertyNames.return_value = image
    image.add.return_value = image
    image.divide.return_value = image
    image.clamp.return_value = image
    image.subtract.return_value = image
    return image


def test_all_indices_registry_has_seven_entries():
    assert len(ALL_INDICES) == 7
    assert set(ALL_INDICES.keys()) == {"NDVI", "NDBI", "MNDWI", "IBI", "BSI", "UI", "ISF"}


@pytest.mark.parametrize(
    "index_cls,required_bands",
    [
        (NDVI, ["nir", "red"]),
        (NDBI, ["swir1", "nir"]),
        (MNDWI, ["green", "swir1"]),
    ],
)
def test_normalized_difference_indices_call_correct_bands(index_cls, required_bands):
    image = _mock_image_with_bands(required_bands)
    index_cls().compute(image)
    image.normalizedDifference.assert_called_with(required_bands)


def test_ndvi_required_bands():
    assert NDVI.required_bands == ["nir", "red"]
    assert NDVI.name == "NDVI"


def test_missing_band_raises():
    from core.exceptions import MissingBandError

    image = _mock_image_with_bands(["red"])  # missing 'nir'
    with pytest.raises(MissingBandError):
        NDVI().compute(image)


def test_ibi_uses_expression_for_savi():
    image = _mock_image_with_bands(["nir", "red", "swir1", "green"])
    IBI().compute(image)
    assert image.expression.called


def test_isf_range_is_clamped_zero_one():
    image = _mock_image_with_bands(["nir", "red", "swir1", "green"])
    ISF().compute(image)
    image.clamp.assert_called_with(0, 1)


def test_bsi_required_bands():
    assert set(BSI.required_bands) == {"swir1", "red", "nir", "blue"}
