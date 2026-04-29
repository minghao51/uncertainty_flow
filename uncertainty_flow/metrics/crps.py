"""Continuous Ranked Probability Score (CRPS) for interval-based predictions."""

import numpy as np
import polars as pl

from ..utils.polars_bridge import to_numpy_series_zero_copy


def crps_score(
    y_true: pl.Series | np.ndarray,
    lower: pl.Series | np.ndarray,
    upper: pl.Series | np.ndarray,
    confidence: float = 0.9,
) -> float:
    """
    Approximate CRPS from a prediction interval.

    Uses the interval approximation:
    CRPS ≈ (upper - lower) / (2 * z_score) * [z_score * (1 - 2*coverage)
             + 2 * phi(z_score) - z_score]

    For a well-calibrated interval, this simplifies to a scaled combination
    of interval width and coverage error.

    Args:
        y_true: True values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        confidence: Confidence level for the interval

    Returns:
        Approximate CRPS score (float). Lower is better.
    """
    if isinstance(y_true, pl.Series):
        y_true = to_numpy_series_zero_copy(y_true)
    if isinstance(lower, pl.Series):
        lower = to_numpy_series_zero_copy(lower)
    if isinstance(upper, pl.Series):
        upper = to_numpy_series_zero_copy(upper)

    alpha = 1 - confidence
    z = _z_score(1 - alpha / 2)

    width = upper - lower
    midpoint = (upper + lower) / 2.0

    sigma = width / (2 * z)

    z_obs = np.where(sigma > 0, (y_true - midpoint) / sigma, 0.0)

    from scipy.stats import norm

    phi_z = norm.pdf(z_obs)

    crps_values = sigma * (z_obs * (2 * norm.cdf(z_obs) - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))

    crps_values = np.where(sigma > 0, crps_values, np.abs(y_true - midpoint))

    return float(np.mean(crps_values))


def _z_score(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))
