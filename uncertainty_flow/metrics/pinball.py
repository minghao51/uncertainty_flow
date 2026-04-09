"""Pinball loss (quantile loss) metric."""

import numpy as np
import polars as pl

from ..utils.exceptions import error_quantile_invalid
from ..utils.polars_bridge import to_numpy_series_zero_copy


def pinball_loss(
    y_true: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
    quantile: float,
) -> float:
    """
    Quantile loss (pinball loss).

    For quantile q, loss = max(q * (y_true - y_pred), (q - 1) * (y_true - y_pred))
    Penalizes over-prediction and under-prediction asymmetrically.

    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Quantile level (e.g., 0.9 for 90th percentile)

    Returns:
        Mean loss across all samples (float)

    Raises:
        ValueError: If quantile is not in (0, 1)

    Examples:
        >>> import polars as pl
        >>> y_true = pl.Series([1, 2, 3, 4, 5])
        >>> y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])
        >>> pinball_loss(y_true, y_pred, 0.5)
        0.4
    """
    if not (0 < quantile < 1):
        error_quantile_invalid(f"quantile must be in (0, 1), got {quantile}")

    # Convert to numpy
    if isinstance(y_true, pl.Series):
        y_true = to_numpy_series_zero_copy(y_true)
    if isinstance(y_pred, pl.Series):
        y_pred = to_numpy_series_zero_copy(y_pred)

    # Compute pinball loss
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)

    return float(np.mean(loss))
