"""Tests for configuration modules (settings, cities, constants, datasets)."""
from __future__ import annotations

import os

import pytest

from config.cities import CityConfig, get_city
from config.constants import (
    LANDSCAPE_METRICS,
    MAX_CLOUD_COVER_PERCENT,
    MIN_SCENES_REQUIRED,
    SEASON_END_MONTH_DAY,
    SEASON_START_MONTH_DAY,
)
from config.datasets import STUDY_YEARS
from config.settings import Settings, get_settings


def test_get_city_returns_hyderabad():
    city = get_city("hyderabad")
    assert isinstance(city, CityConfig)
    assert city.epsg == 32644
    assert city.name == "Hyderabad"


def test_get_city_case_insensitive():
    assert get_city("HYDERABAD").name == "Hyderabad"


def test_get_city_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_city("nonexistent_city")


def test_study_years_are_five_year_intervals():
    assert STUDY_YEARS == [2000, 2005, 2010, 2015, 2020, 2025]


def test_season_window_format():
    assert SEASON_START_MONTH_DAY == "02-01"
    assert SEASON_END_MONTH_DAY == "05-31"


def test_cloud_and_scene_thresholds_are_sane():
    assert 0 < MAX_CLOUD_COVER_PERCENT <= 100
    assert MIN_SCENES_REQUIRED >= 1


def test_landscape_metrics_nonempty():
    assert len(LANDSCAPE_METRICS) > 0


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("CITY_KEY", raising=False)
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.city_key == "hyderabad"
    assert settings.city.name == "Hyderabad"


def test_settings_output_dir_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    settings = get_settings()
    out = settings.output_dir("indices", "2020")
    assert out.exists()
    assert out.is_dir()
