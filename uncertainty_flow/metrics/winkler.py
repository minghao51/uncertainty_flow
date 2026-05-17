"""Winkler score for prediction intervals."""

import numpy as np
import polars as pl

from ..utils.exceptions import QuantileError
from ..utils.polars_bridge import as_numpy, validate_bounds


def winkler_score(
    y_true: pl.Series | np.ndarray,
    lower: pl.Series | np.ndarray,
    upper: pl.Series | np.ndarray,
    confidence: float,
) -> float:
    """
    Winkler score for prediction intervals.

    Penalizes:
    - Interval width (wider intervals = higher penalty)
    - Misses (if y_true outside interval, penalty proportional to distance)

    Lower is better.

    Args:
        y_true: True values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        confidence: Confidence level (e.g., 0.9 for 90% interval)

    Returns:
        Mean Winkler score across all samples (float)

    Raises:
        ValueError: If confidence is not in (0, 1) or if bounds are invalid

    Examples:
        >>> import polars as pl
        >>> y_true = pl.Series([1, 2, 3, 4, 5])
        >>> lower = pl.Series([0.5, 1.5, 2.5, 3.5, 4.5])
        >>> upper = pl.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        >>> winkler_score(y_true, lower, upper, 0.9)
        1.0
    """
    if not (0 < confidence < 1):
        raise QuantileError(f"confidence must be in (0, 1), got {confidence}")

    y_true, lower, upper = as_numpy(y_true, lower, upper)

    validate_bounds(lower, upper)

    alpha = 1 - confidence

    width_penalty = upper - lower

    miss_penalty = np.zeros_like(y_true)
    below_mask = y_true < lower
    above_mask = y_true > upper

    miss_penalty[below_mask] = (2 / alpha) * (lower[below_mask] - y_true[below_mask])
    miss_penalty[above_mask] = (2 / alpha) * (y_true[above_mask] - upper[above_mask])

    score = width_penalty + miss_penalty

    return float(np.mean(score))
