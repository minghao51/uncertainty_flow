"""Metrics for evaluating probabilistic predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import polars as pl

from .calibration import calibration_error
from .comparison import diebold_mariano_test, model_confidence_set, skill_score
from .coverage import coverage_score
from .crps import crps_quantile, crps_score
from .log_score import log_score, log_score_kde, log_score_pooled
from .multivariate import energy_score, variogram_score
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
    "diebold_mariano_test",
    "energy_score",
    "log_score",
    "log_score_kde",
    "log_score_pooled",
    "mae_score",
    "model_confidence_set",
    "pinball_loss",
    "rmse_score",
    "score",
    "skill_score",
    "variogram_score",
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
    "log_score",
    "energy_score",
    "variogram_score",
}


def _median_for_target(point, target: str) -> pl.Series:
    """Extract median values for a target from either univariate or multivariate output."""
    if isinstance(point, pl.DataFrame):
        return point[target]
    return point


def score(
    pred: DistributionPrediction,
    y_true,
    metric: str | Callable,
    **kwargs,
) -> float | dict[str, float]:
    """Unified metric entry point.

    Dispatches to the correct metric function based on ``metric`` name,
    extracting the right inputs from the ``DistributionPrediction`` object.

    Args:
        pred: A ``DistributionPrediction`` from any model's ``.predict()``.
        y_true: True values (Polars Series/DataFrame or numpy array).
        metric: One of ``"crps"``, ``"pinball"``, ``"coverage"``,
            ``"winkler"``, ``"mae"``, ``"rmse"``, ``"calibration_error"``,
            or a callable ``fn(pred, y_true) -> float``.
        **kwargs: Extra keyword arguments forwarded to the underlying metric.

    Returns:
        Scalar for univariate, or ``{target: value}`` dict for multivariate.
    """
    if callable(metric):
        return float(metric(pred, y_true, **kwargs))

    if metric not in _METRIC_NAMES:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: {sorted(_METRIC_NAMES)}")

    if metric == "crps":
        return pred.crps(y_true)

    if metric == "log_score":
        return pred.log_score(y_true, family=kwargs.get("family", "auto"))

    if metric == "energy_score":
        return pred.energy_score(
            y_true,
            n_samples=kwargs.get("n_samples", 1000),
            random_state=kwargs.get("random_state"),
        )

    if metric == "variogram_score":
        return variogram_score(
            pred,
            y_true,
            n_samples=kwargs.get("n_samples", 1000),
            p=kwargs.get("p", 0.5),
            random_state=kwargs.get("random_state"),
        )

    y_arr = pred._coerce_y_true(y_true)
    confidence = kwargs.get("confidence", 0.9)
    targets = pred._targets
    n_targets = len(targets)
    results: dict[str, float] = {}

    median_result = pred.median() if metric in ("mae", "rmse") else None

    for t_idx, target in enumerate(targets):
        y_col = y_arr[:, t_idx] if y_arr.ndim == 2 else y_arr
        median_col = (
            _median_for_target(median_result, target) if median_result is not None else None
        )

        if metric == "mae":
            results[target] = float(mae_score(y_col, median_col.to_numpy()))
            continue

        if metric == "rmse":
            results[target] = float(rmse_score(y_col, median_col.to_numpy()))
            continue

        if metric == "pinball":
            total = 0.0
            for level in pred._levels:
                q_df = pred.quantile(float(level))
                if n_targets == 1:
                    q_vals = q_df.to_numpy().flatten()
                else:
                    q_vals = q_df[f"{target}_q_{level:.3f}"].to_numpy()
                total += pinball_loss(y_col, q_vals, float(level))
            results[target] = total / len(pred._levels)
            continue

        lower_s, upper_s = pred.interval_bounds(confidence, target=target)
        lower = lower_s.to_numpy()
        upper = upper_s.to_numpy()

        if metric == "coverage":
            results[target] = float(coverage_score(y_col, lower, upper))
        elif metric == "winkler":
            results[target] = float(winkler_score(y_col, lower, upper, confidence))
        elif metric == "calibration_error":
            results[target] = float(calibration_error(y_col, lower, upper, confidence))

    if n_targets == 1:
        return results[targets[0]]
    return results
