"""Shared pytest fixtures for uncertainty_flow tests."""

import numpy as np
import polars as pl
import pytest


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
