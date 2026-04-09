"""Coverage score metric."""

import numpy as np
import polars as pl

from ..utils.exceptions import error_invalid_data
from ..utils.polars_bridge import to_numpy_series_zero_copy


def coverage_score(
    y_true: pl.Series | np.ndarray,
    lower: pl.Series | np.ndarray,
    upper: pl.Series | np.ndarray,
) -> float:
    """
    Fraction of true values that fall within the prediction interval.

    Args:
        y_true: True values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval

    Returns:
        Fraction of values within interval (float in [0, 1])

    Raises:
        ValueError: If bounds are invalid

    Examples:
        >>> import polars as pl
        >>> y_true = pl.Series([1, 2, 3, 4, 5])
        >>> lower = pl.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        >>> upper = pl.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        >>> coverage_score(y_true, lower, upper)
        0.6
    """
    # Convert to numpy
    if isinstance(y_true, pl.Series):
        y_true = to_numpy_series_zero_copy(y_true)
    if isinstance(lower, pl.Series):
        lower = to_numpy_series_zero_copy(lower)
    if isinstance(upper, pl.Series):
        upper = to_numpy_series_zero_copy(upper)

    # Validate bounds
    if np.any(lower > upper):
        error_invalid_data("lower bound must be <= upper bound")

    # Count how many values are within the interval
    within_interval = (y_true >= lower) & (y_true <= upper)

    return float(np.mean(within_interval))
