"""Point prediction metrics: MAE and RMSE."""

import numpy as np
import polars as pl

from ..utils.polars_bridge import as_numpy


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
    y_true, y_pred = as_numpy(y_true, y_pred)
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
    y_true, y_pred = as_numpy(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
