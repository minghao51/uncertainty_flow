"""Shared pytest fixtures and auto-mark hook for uncertainty_flow tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

try:
    import matplotlib
except ImportError:
    matplotlib = None

_INTEGRATION_DIRS = {
    "models",
    "wrappers",
    "analysis",
    "causal",
    "counterfactual",
    "decomposition",
    "metrics",
    "multivariate",
    "multimodal",
    "risk",
    "viz",
    "bayesian",
}

_INTEGRATION_ROOT_FILES = {
    "test_package_integration.py",
    "test_base_quantile.py",
}

_INTEGRATION_CORE_FILES = {
    "test_persistence.py",
    "test_distribution.py",
}

_INTEGRATION_UTILS_FILES = {
    "test_calibration_report.py",
}

_SLOW_FILES = {
    "test_conformal_ts.py",
    "test_counterfactual.py",
    "test_deep_quantile_torch.py",
    "test_numpyro_model.py",
    "test_persistence.py",
}

_OPTIONAL_FILES = {
    "test_deep_quantile_torch.py",
    "test_numpyro_model.py",
}


def _marker_names_for_nodeid(nodeid: str) -> set[str]:
    """Return the implicit pytest markers for a collected test node id."""
    path = nodeid.split("::", 1)[0]
    parts = path.split("/")
    file_name = parts[-1] if parts else ""

    markers: set[str] = set()

    if file_name in _OPTIONAL_FILES:
        markers.add("optional")

    is_integration = False

    if len(parts) >= 3:
        subdir = parts[-2]
        if subdir in _INTEGRATION_DIRS:
            is_integration = True
    elif len(parts) == 2 and file_name in _INTEGRATION_ROOT_FILES:
        is_integration = True

    if file_name in _INTEGRATION_CORE_FILES or file_name in _INTEGRATION_UTILS_FILES:
        is_integration = True

    if is_integration:
        markers.add("integration")
    else:
        markers.add("unit")

    if file_name in _SLOW_FILES:
        markers.add("slow")

    return markers


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for marker_name in _marker_names_for_nodeid(item.nodeid):
            item.add_marker(getattr(pytest.mark, marker_name))


@pytest.fixture
def time_series_data():
    """Create extended time series DataFrame for testing (150 rows)."""
    rng = np.random.default_rng(42)
    n = 150
    return pl.DataFrame(
        {
            "date": range(n),
            "price": [10 + i * 0.5 + np.sin(i / 3) + rng.standard_normal() * 0.5 for i in range(n)],
            "volume": [100 + i * 2 + np.cos(i / 2) + rng.standard_normal() * 5 for i in range(n)],
        }
    )


@pytest.fixture
def univariate_time_series():
    """Create univariate time series DataFrame (150 rows)."""
    rng = np.random.default_rng(42)
    n = 150
    return pl.DataFrame(
        {
            "date": range(n),
            "target": [
                10 + i * 0.5 + np.sin(i / 3) + rng.standard_normal() * 0.5 for i in range(n)
            ],
        }
    )
