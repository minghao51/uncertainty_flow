"""Regression tests for pytest auto-marking conventions."""

from __future__ import annotations

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
