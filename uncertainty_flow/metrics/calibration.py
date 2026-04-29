"""Calibration error metric."""

import numpy as np
import polars as pl

from ..utils.polars_bridge import to_numpy_series_zero_copy


def calibration_error(
    y_true: pl.Series | np.ndarray,
    lower: pl.Series | np.ndarray,
    upper: pl.Series | np.ndarray,
    nominal_coverage: float = 0.9,
) -> float:
    """
    Absolute deviation of empirical coverage from nominal coverage.

    Args:
        y_true: True values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        nominal_coverage: Target coverage level (e.g. 0.9)

    Returns:
        Absolute calibration error (float). Lower is better. 0 = perfectly calibrated.
    """
    if isinstance(y_true, pl.Series):
        y_true = to_numpy_series_zero_copy(y_true)
    if isinstance(lower, pl.Series):
        lower = to_numpy_series_zero_copy(lower)
    if isinstance(upper, pl.Series):
        upper = to_numpy_series_zero_copy(upper)

    empirical_coverage = np.mean((y_true >= lower) & (y_true <= upper))

    return float(np.abs(empirical_coverage - nominal_coverage))
