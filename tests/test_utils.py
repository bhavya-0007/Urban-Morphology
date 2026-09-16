"""Tests for core.utils generic helpers."""
from __future__ import annotations

import time

import pytest

from core.exceptions import UrbanMorphologyError
from core.utils import Timer, chunked, ensure_dir, retry, safe_filename, season_date_range


def test_season_date_range():
    start, end = season_date_range(2020, "02-01", "05-31")
    assert start == "2020-02-01"
    assert end == "2020-05-31"


def test_ensure_dir_creates_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.exists()
    assert result.is_dir()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Landsat Composite 2020!", "landsat_composite_2020"),
        ("  spaced  out  ", "spaced_out"),
        ("already_clean", "already_clean"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_chunked_splits_evenly():
    assert chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunked_splits_with_remainder():
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_retry_succeeds_after_transient_failures():
    calls = {"count": 0}

    @retry(times=3, delay_seconds=0.01, exceptions=(ValueError,))
    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 2


def test_retry_raises_after_exhausting_attempts():
    @retry(times=2, delay_seconds=0.01, exceptions=(UrbanMorphologyError,))
    def always_fails():
        raise UrbanMorphologyError("permanent failure")

    with pytest.raises(UrbanMorphologyError):
        always_fails()


def test_timer_measures_elapsed_time():
    with Timer("test-block") as t:
        time.sleep(0.05)
    assert t.elapsed_seconds >= 0.05
