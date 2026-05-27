"""Standalone scoring functions for DistributionPrediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..utils.exceptions import InvalidDataError

if TYPE_CHECKING:
    from .distribution import DistributionPrediction


def crps_score(
    prediction: DistributionPrediction,
    y_true: pl.Series | pl.DataFrame | np.ndarray,
) -> float | dict[str, float]:
    from ..metrics.crps import crps_quantile

    if prediction._n_quantiles < 2:
        raise InvalidDataError(
            "CRPS requires at least 2 quantile levels, "
            f"but DistributionPrediction has {prediction._n_quantiles}"
        )

    y_arr = prediction._coerce_y_true(y_true)

    if len(prediction._targets) == 1:
        return crps_quantile(y_arr, prediction._quantiles, prediction._levels)

    result = {}
    for t_idx, target in enumerate(prediction._targets):
        q_slice = prediction._quantile_slice(t_idx)
        result[target] = crps_quantile(
            prediction._target_truth(y_arr, t_idx),
            q_slice,
            prediction._levels,
        )
    return result


def log_score(
    prediction: DistributionPrediction,
    y_true: pl.Series | pl.DataFrame | np.ndarray,
    family: str = "auto",
) -> float | dict[str, float]:
    from ..metrics.log_score import log_score as _log_score

    y_arr = prediction._coerce_y_true(y_true)

    if len(prediction._targets) == 1:
        return _log_score(
            y_arr,
            prediction._quantiles[:, : prediction._n_quantiles],
            prediction._levels,
            family=family,
        )

    result = {}
    for t_idx, target in enumerate(prediction._targets):
        result[target] = _log_score(
            prediction._target_truth(y_arr, t_idx),
            prediction._quantile_slice(t_idx),
            prediction._levels,
            family=family,
        )
    return result


def energy_score(
    prediction: DistributionPrediction,
    y_true: pl.Series | pl.DataFrame | np.ndarray,
    n_samples: int = 1000,
    random_state: int | None = None,
) -> float:
    from ..metrics.multivariate import energy_score as _energy_score

    return _energy_score(prediction, y_true, n_samples=n_samples, random_state=random_state)


def variogram_score(
    prediction: DistributionPrediction,
    y_true: pl.Series | pl.DataFrame | np.ndarray,
    n_samples: int = 1000,
    p: float = 0.5,
    random_state: int | None = None,
) -> float:
    from ..metrics.multivariate import variogram_score as _variogram_score

    return _variogram_score(prediction, y_true, n_samples=n_samples, p=p, random_state=random_state)
