"""Regression tests for pytest auto-marking conventions."""

from __future__ import annotations

import numpy as np
import polars as pl

from tests.conftest import _marker_names_for_nodeid


def test_optional_marker_matches_filename_special_cases():
    """Optional files should keep their filename-based auto-markers."""
    marker_names = _marker_names_for_nodeid(
        "tests/models/test_deep_quantile_torch.py::TestThing::test_case"
    )

    assert "optional" in marker_names
    assert "integration" in marker_names
    assert "slow" in marker_names


def test_core_filename_special_case_keeps_integration_marker():
    """Core slow files should keep integration markers after stripping selectors."""
    marker_names = _marker_names_for_nodeid(
        "tests/core/test_persistence.py::TestModelRoundTrip::test_quantile_forest_round_trip"
    )

    assert "integration" in marker_names
    assert "slow" in marker_names
    assert "unit" not in marker_names


def test_non_integration_file_defaults_to_unit():
    marker_names = _marker_names_for_nodeid("tests/test_config.py::test_something")

    assert "unit" in marker_names
    assert "integration" not in marker_names
    assert "slow" not in marker_names


def test_integration_root_file_is_classified_as_integration():
    marker_names = _marker_names_for_nodeid("tests/test_full_workflow.py::test_end_to_end")

    assert "integration" in marker_names
    assert "unit" not in marker_names


def test_integration_subdir_file_is_classified_as_integration():
    marker_names = _marker_names_for_nodeid("tests/wrappers/test_conformal.py::test_fit")

    assert "integration" in marker_names
    assert "unit" not in marker_names


def test_time_series_fixture_shape_and_columns(time_series_data: pl.DataFrame):
    assert time_series_data.shape == (150, 3)
    assert time_series_data.columns == ["date", "price", "volume"]


def test_time_series_fixture_is_deterministic(time_series_data: pl.DataFrame):
    rng = np.random.default_rng(42)
    expected_all_price = [
        10 + i * 0.5 + np.sin(i / 3) + rng.standard_normal() * 0.5 for i in range(150)
    ]
    expected_all_volume = [
        100 + i * 2 + np.cos(i / 2) + rng.standard_normal() * 5 for i in range(150)
    ]

    assert time_series_data["price"].head(3).to_list() == expected_all_price[:3]
    assert time_series_data["volume"].head(3).to_list() == expected_all_volume[:3]


def test_univariate_fixture_shape_and_columns(univariate_time_series: pl.DataFrame):
    assert univariate_time_series.shape == (150, 2)
    assert univariate_time_series.columns == ["date", "target"]
