"""Point prediction metrics: MAE and RMSE."""

import numpy as np
import polars as pl

from ..utils.polars_bridge import to_numpy_series_zero_copy


def mae_score(
    y_true: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values (point predictions, e.g. median)

    Returns:
        Mean absolute error (float). Lower is better.
    """
    if isinstance(y_true, pl.Series):
        y_true = to_numpy_series_zero_copy(y_true)
    if isinstance(y_pred, pl.Series):
        y_pred = to_numpy_series_zero_copy(y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_score(
    y_true: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values (point predictions, e.g. median)

    Returns:
        Root mean squared error (float). Lower is better.
    """
    if isinstance(y_true, pl.Series):
        y_true = to_numpy_series_zero_copy(y_true)
    if isinstance(y_pred, pl.Series):
        y_pred = to_numpy_series_zero_copy(y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
