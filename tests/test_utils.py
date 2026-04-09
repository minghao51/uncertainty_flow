"""Shared test utilities and helper functions."""

from __future__ import annotations

import numpy as np
import polars as pl


def create_test_distribution(
    n_samples: int = 100,
    n_targets: int = 1,
    quantile_levels: list[float] | None = None,
) -> object:
    """
    Create a test DistributionPrediction with synthetic data.

    Args:
        n_samples: Number of samples to generate
        n_targets: Number of target variables
        quantile_levels: List of quantile levels to use

    Returns:
        DistributionPrediction object for testing
    """
    from uncertainty_flow.core.distribution import DistributionPrediction

    if quantile_levels is None:
        quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

    np.random.seed(42)
    n_quantiles = len(quantile_levels)

    # Create synthetic quantile matrix
    quantile_matrix = np.random.randn(n_samples, n_targets * n_quantiles) * 10 + 50

    # Ensure monotonicity (sort quantiles within each row and target)
    for i in range(n_samples):
        for t in range(n_targets):
            start_idx = t * n_quantiles
            end_idx = start_idx + n_quantiles
            quantile_matrix[i, start_idx:end_idx] = np.sort(quantile_matrix[i, start_idx:end_idx])

    target_names = [f"target_{i}" for i in range(n_targets)]

    return DistributionPrediction(
        quantile_matrix=quantile_matrix,
        quantile_levels=quantile_levels,
        target_names=target_names,
    )


def assert_interval_properties(
    interval: pl.DataFrame,
    target_name: str = "target",
    min_width: float = 0.0,
    max_width: float = 1000.0,
) -> None:
    """
    Assert interval has reasonable properties.

    Args:
        interval: Interval DataFrame with lower/upper columns
        target_name: Name of the target column
        min_width: Minimum acceptable interval width
        max_width: Maximum acceptable interval width

    Raises:
        AssertionError: If interval properties are invalid
    """
    if "lower" in interval.columns:
        lower_col = "lower"
        upper_col = "upper"
    else:
        lower_col = f"{target_name}_lower"
        upper_col = f"{target_name}_upper"

    lower = interval[lower_col].to_numpy()
    upper = interval[upper_col].to_numpy()

    # Check bounds
    assert np.all(lower <= upper), "Found lower bound > upper bound"

    # Check widths
    widths = upper - lower
    assert np.all(widths >= min_width), f"Found interval width < {min_width}"
    assert np.all(widths <= max_width), f"Found interval width > {max_width}"

    # Check for NaN/Inf
    assert np.all(np.isfinite(lower)), "Found non-finite values in lower bound"
    assert np.all(np.isfinite(upper)), "Found non-finite values in upper bound"


def compute_empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Compute empirical coverage score.

    Args:
        y_true: True values
        lower: Lower bound values
        upper: Upper bound values

    Returns:
        Fraction of values within the interval
    """
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def create_bivariate_residuals(n_samples: int = 1000, correlation: float = 0.7) -> np.ndarray:
    """
    Create synthetic bivariate residuals with specified correlation.

    Args:
        n_samples: Number of samples to generate
        correlation: Desired correlation between variables

    Returns:
        Array of shape (n_samples, 2) with correlated residuals
    """
    np.random.seed(42)

    # Create correlated variables
    x = np.random.randn(n_samples)
    y = correlation * x + np.sqrt(1 - correlation**2) * np.random.randn(n_samples)

    return np.column_stack([x, y])


def create_time_series_with_pattern(
    n: int = 200,
    trend: float = 0.1,
    seasonality: bool = True,
    noise_std: float = 0.5,
) -> pl.DataFrame:
    """
    Create a time series with trend, seasonality, and noise.

    Args:
        n: Number of time points
        trend: Linear trend coefficient
        seasonality: Whether to add seasonal component
        noise_std: Standard deviation of noise

    Returns:
        DataFrame with date and value columns
    """
    np.random.seed(42)

    dates = range(n)
    values = [trend * i for i in range(n)]

    if seasonality:
        values = [v + 5 * np.sin(i / 10) for i, v in enumerate(values)]

    values = [v + np.random.randn() * noise_std for v in values]

    return pl.DataFrame({"date": dates, "value": values})
