"""Metrics for evaluating probabilistic predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .calibration import calibration_error
from .coverage import coverage_score
from .crps import crps_quantile, crps_score
from .pinball import pinball_loss
from .point import mae_score, rmse_score
from .winkler import winkler_score

if TYPE_CHECKING:
    from ..core.distribution import DistributionPrediction

__all__ = [
    "calibration_error",
    "coverage_score",
    "crps_quantile",
    "crps_score",
    "mae_score",
    "pinball_loss",
    "rmse_score",
    "score",
    "winkler_score",
]

_METRIC_NAMES = {
    "crps",
    "pinball",
    "coverage",
    "winkler",
    "mae",
    "rmse",
    "calibration_error",
}


def score(
    pred: DistributionPrediction,
    y_true,
    metric: str | Callable,
    **kwargs,
) -> float | dict[str, float]:
    """
    Unified metric entry point.

    Dispatches to the correct metric function based on ``metric`` name,
    extracting the right inputs (intervals, quantiles, point estimates)
    from the ``DistributionPrediction`` object.

    For multivariate predictions (2+ targets), returns a ``{target: value}``
    dict instead of a scalar float.

    Args:
        pred: A ``DistributionPrediction`` from any model's ``.predict()``.
        y_true: True values (Polars Series/DataFrame or numpy array).
        metric: One of ``"crps"``, ``"pinball"``, ``"coverage"``,
            ``"winkler"``, ``"mae"``, ``"rmse"``, ``"calibration_error"``,
            or a callable ``fn(pred, y_true) -> float``.
        **kwargs: Extra keyword arguments forwarded to the underlying metric.

    Returns:
        Scalar metric value (float) for univariate predictions, or
        ``{target: value}`` dict for multivariate predictions.

    Examples:
        >>> from uncertainty_flow.metrics import score
        >>> crps = score(pred, y_true, metric="crps")
        >>> cov = score(pred, y_true, metric="coverage")
    """
    if callable(metric):
        return float(metric(pred, y_true, **kwargs))

    if metric not in _METRIC_NAMES:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: {sorted(_METRIC_NAMES)}")

    confidence = kwargs.get("confidence", 0.9)
    n_targets = len(pred._targets)

    if metric == "crps":
        return pred.crps(y_true)

    y_arr = pred._coerce_y_true(y_true)

    if n_targets == 1:
        return _score_univariate(pred, y_arr, metric, confidence)
    return _score_multivariate(pred, y_arr, metric, confidence)


def _score_univariate(
    pred: DistributionPrediction,
    y_arr: np.ndarray,
    metric: str,
    confidence: float,
) -> float:
    if metric == "mae":
        point = pred.median()
        y_pred = point.to_numpy() if hasattr(point, "to_numpy") else np.asarray(point)
        return float(mae_score(y_arr, y_pred))

    if metric == "rmse":
        point = pred.median()
        y_pred = point.to_numpy() if hasattr(point, "to_numpy") else np.asarray(point)
        return float(rmse_score(y_arr, y_pred))

    interval_df = pred.interval(confidence)
    lower = interval_df["lower"].to_numpy()
    upper = interval_df["upper"].to_numpy()

    if metric == "coverage":
        return float(coverage_score(y_arr, lower, upper))
    if metric == "winkler":
        return float(winkler_score(y_arr, lower, upper, confidence))
    if metric == "calibration_error":
        return float(calibration_error(y_arr, lower, upper, confidence))

    if metric == "pinball":
        total = 0.0
        for level in pred._levels:
            q_vals = pred.quantile(float(level)).to_numpy().flatten()
            total += pinball_loss(y_arr, q_vals, float(level))
        return total / len(pred._levels)

    raise ValueError(f"Unhandled metric: {metric!r}")


def _score_multivariate(
    pred: DistributionPrediction,
    y_arr: np.ndarray,
    metric: str,
    confidence: float,
) -> dict[str, float]:
    results: dict[str, float] = {}

    for t_idx, target in enumerate(pred._targets):
        if y_arr.ndim == 2:
            y_col = y_arr[:, t_idx]
        else:
            y_col = y_arr

        if metric == "mae":
            point = pred.median()
            y_pred = (
                point[target].to_numpy()
                if hasattr(point, "to_numpy")
                else np.asarray(point)[:, t_idx]
            )
            results[target] = float(mae_score(y_col, y_pred))
            continue

        if metric == "rmse":
            point = pred.median()
            y_pred = (
                point[target].to_numpy()
                if hasattr(point, "to_numpy")
                else np.asarray(point)[:, t_idx]
            )
            results[target] = float(rmse_score(y_col, y_pred))
            continue

        if metric == "pinball":
            total = 0.0
            for level in pred._levels:
                q_df = pred.quantile(float(level))
                q_vals = q_df[f"{target}_q_{level:.3f}"].to_numpy()
                total += pinball_loss(y_col, q_vals, float(level))
            results[target] = total / len(pred._levels)
            continue

        interval_df = pred.interval(confidence)
        lower = interval_df[f"{target}_lower"].to_numpy()
        upper = interval_df[f"{target}_upper"].to_numpy()

        if metric == "coverage":
            results[target] = float(coverage_score(y_col, lower, upper))
        elif metric == "winkler":
            results[target] = float(winkler_score(y_col, lower, upper, confidence))
        elif metric == "calibration_error":
            results[target] = float(calibration_error(y_col, lower, upper, confidence))

    return results
