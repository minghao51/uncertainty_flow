"""Shared auto-tuning helpers for model classes."""

from __future__ import annotations

from itertools import product
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from ..metrics import coverage_score, winkler_score
from .polars_bridge import to_numpy_series


def candidate_values(current: Any, defaults: list[Any]) -> list[Any]:
    """Return a stable unique candidate list including the current value."""
    values = [current, *defaults]
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def valid_calibration_candidates(
    n_rows: int,
    current: float,
    defaults: list[float],
    min_samples: int = 20,
) -> list[float]:
    """Return only calibration sizes that keep the calibration split large enough."""
    candidates = candidate_values(current, defaults)
    valid = [value for value in candidates if int(n_rows * value) >= min_samples]
    return valid or [max(current, min(0.5, min_samples / max(n_rows, 1)))]


def estimator_param_candidates(base_model: BaseEstimator) -> list[dict[str, Any]]:
    """Build a lightweight tuning grid from supported estimator params."""
    params = base_model.get_params(deep=False)
    grid: dict[str, list[Any]] = {}

    if "n_estimators" in params:
        grid["n_estimators"] = candidate_values(params["n_estimators"], [20, 30, 50])

    if "max_depth" in params and params["max_depth"] is not None:
        current = params["max_depth"]
        grid["max_depth"] = candidate_values(current, [3, 5, 8])

    if "learning_rate" in params:
        current = params["learning_rate"]
        grid["learning_rate"] = candidate_values(current, [0.03, 0.05, 0.1])

    if not grid:
        return [{}]

    keys = list(grid.keys())
    return [
        dict(zip(keys, values, strict=False)) for values in product(*(grid[key] for key in keys))
    ]


def score_distribution_prediction(
    pred,
    actuals: pl.Series | pl.DataFrame,
    target_names: list[str],
    confidence: float = 0.9,
) -> float:
    """
    Score a prediction using median error, coverage, sharpness, and Winkler score.

    Lower is better.
    """
    interval = pred.interval(confidence)
    mean_pred = pred.median()

    scores = []
    for target in target_names:
        if isinstance(actuals, pl.DataFrame):
            y_true = to_numpy_series(actuals[target])
        else:
            y_true = actuals.to_numpy()

        if isinstance(mean_pred, pl.DataFrame):
            median = to_numpy_series(mean_pred[target])
            lower = to_numpy_series(interval[f"{target}_lower"])
            upper = to_numpy_series(interval[f"{target}_upper"])
        else:
            median = mean_pred.to_numpy()
            lower = to_numpy_series(interval["lower"])
            upper = to_numpy_series(interval["upper"])

        n = min(len(y_true), len(median))
        y_true = y_true[-n:]
        median = median[-n:]
        lower = lower[-n:]
        upper = upper[-n:]

        mae = float(np.mean(np.abs(y_true - median)))
        coverage = float(coverage_score(y_true, lower, upper))
        sharpness = float(np.mean(upper - lower))
        winkler = float(winkler_score(y_true, lower, upper, confidence=confidence))

        coverage_error = abs(coverage - confidence)
        if coverage_error > 0.15:
            coverage_error *= 10

        scores.append(mae + coverage_error * 0.5 + sharpness * 0.1 + winkler * 0.1)

    return float(np.mean(scores))


def build_tune_splits(
    data: pl.DataFrame,
    *,
    task_type: Literal["tabular", "time_series"],
    random_state: int | None = None,
    hybrid_mode: bool = False,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Build evaluation splits for auto-tuning, with CV averaging when inner splits exist."""
    from .split import select_validation_plan

    plan = select_validation_plan(
        data,
        task_type=task_type,
        random_state=random_state,
        holdout_fraction=0.2,
        hybrid_mode=hybrid_mode,
    )
    return plan.inner_splits if plan.inner_splits else [plan.outer_split]
