"""Shared pytest fixtures for uncertainty_flow tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "target": [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5],
        }
    )


@pytest.fixture
def sample_time_series():
    """Create a sample time series DataFrame for testing."""
    return pl.DataFrame(
        {
            "date": range(20),
            "price": [10 + i + 0.5 * np.sin(i / 3) for i in range(20)],
            "volume": [100 + 10 * i + 5 * np.cos(i / 2) for i in range(20)],
        }
    )


@pytest.fixture
def random_state():
    """Default random state for reproducibility."""
    return 42


# ============================================================================
# Extended Fixtures for Larger Test Suites
# ============================================================================


@pytest.fixture
def time_series_data():
    """Create extended time series DataFrame for testing (150 rows)."""
    np.random.seed(42)
    n = 150
    return pl.DataFrame(
        {
            "date": range(n),
            "price": [10 + i * 0.5 + np.sin(i / 3) + np.random.randn() * 0.5 for i in range(n)],
            "volume": [100 + i * 2 + np.cos(i / 2) + np.random.randn() * 5 for i in range(n)],
        }
    )


@pytest.fixture
def univariate_time_series():
    """Create univariate time series DataFrame (150 rows)."""
    np.random.seed(42)
    n = 150
    return pl.DataFrame(
        {
            "date": range(n),
            "target": [10 + i * 0.5 + np.sin(i / 3) + np.random.randn() * 0.5 for i in range(n)],
        }
    )


@pytest.fixture
def multivariate_time_series():
    """Create multivariate time series with multiple targets."""
    np.random.seed(42)
    n = 200
    return pl.DataFrame(
        {
            "date": range(n),
            "price": [100 + i * 0.3 + np.random.randn() * 2 for i in range(n)],
            "demand": [50 + i * 0.2 + np.random.randn() * 1 for i in range(n)],
            "inventory": [200 + i * 0.1 + np.random.randn() * 3 for i in range(n)],
        }
    )


@pytest.fixture
def sample_quantile_matrix():
    """Create sample quantile matrix for testing."""
    return np.array(
        [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
        ]
    )


@pytest.fixture
def base_model():
    """Create base sklearn model for testing."""
    return GradientBoostingRegressor(n_estimators=30, random_state=42)


@pytest.fixture
def quantile_levels():
    """Standard quantile levels for testing."""
    return [0.1, 0.25, 0.5, 0.75, 0.9]


# ============================================================================
# Parameterized Fixtures
# ============================================================================


@pytest.fixture(params=[0.8, 0.9, 0.95])
def confidence_level(request):
    """Parameterized confidence levels for testing."""
    return request.param


@pytest.fixture(params=[20, 50, 100, 200])
def sample_size(request):
    """Parameterized sample sizes for testing."""
    return request.param
