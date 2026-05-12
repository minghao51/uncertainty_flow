"""Continuous Ranked Probability Score (CRPS) for probabilistic predictions."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from ..utils.polars_bridge import as_numpy


def crps_quantile(
    y_true: np.ndarray,
    quantile_matrix: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    """
    Exact CRPS from quantile predictions via the quantile-score decomposition.

    Uses the closed-form decomposition:

        CRPS = 2 * Σⱼ wⱼ * [𝟙(y < qⱼ) - τⱼ] * (qⱼ - y)

    where wⱼ = (τⱼ₊₁ - τⱼ₋₁) / 2 are trapezoidal quadrature weights.

    Reference: Laio & Tamea (2007); also used by ``properscoring`` and
    ``scoringrules`` packages.

    Args:
        y_true: (n,) array of true values.
        quantile_matrix: (n, k) array of predicted quantile values per sample.
        quantile_levels: (k,) array of quantile levels in (0, 1), strictly increasing.

    Returns:
        Mean CRPS across all samples. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    quantile_matrix = np.asarray(quantile_matrix, dtype=np.float64)
    quantile_levels = np.asarray(quantile_levels, dtype=np.float64)

    k = len(quantile_levels)
    if k < 2:
        raise ValueError("quantile_levels must have at least 2 elements")

    if np.any(np.diff(quantile_matrix, axis=1) < 0):
        warnings.warn(
            "quantile_matrix contains non-monotone columns (quantile crossing). "
            "CRPS results may be unreliable. Ensure quantile values are sorted "
            "per sample.",
            UserWarning,
            stacklevel=2,
        )

    weights = np.empty(k)
    weights[0] = (quantile_levels[1] - quantile_levels[0]) / 2.0
    weights[-1] = (quantile_levels[-1] - quantile_levels[-2]) / 2.0
    if k > 2:
        weights[1:-1] = (quantile_levels[2:] - quantile_levels[:-2]) / 2.0

    indicator = (y_true[:, None] < quantile_matrix).astype(np.float64)
    diff = indicator - quantile_levels[None, :]
    residual = quantile_matrix - y_true[:, None]

    crps_per_sample = 2.0 * np.sum(weights[None, :] * diff * residual, axis=1)

    return float(np.mean(crps_per_sample))


def crps_score(
    y_true: pl.Series | np.ndarray,
    lower: pl.Series | np.ndarray,
    upper: pl.Series | np.ndarray,
    confidence: float = 0.9,
) -> float:
    """
    Approximate CRPS from a prediction interval (Gaussian assumption).

    .. deprecated::
        Use :func:`crps_quantile` or ``DistributionPrediction.crps(y_true)``
        instead. This function will be removed in v0.3.0.

    Args:
        y_true: True values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        confidence: Confidence level for the interval

    Returns:
        Approximate CRPS score (float). Lower is better.
    """
    warnings.warn(
        "crps_score() uses a Gaussian approximation and will be removed in "
        "v0.3.0. Use crps_quantile() or DistributionPrediction.crps() instead.",
        FutureWarning,
        stacklevel=2,
    )

    from scipy.stats import norm

    y_true, lower, upper = as_numpy(y_true, lower, upper)

    alpha = 1 - confidence
    z = float(norm.ppf(1 - alpha / 2))

    width = upper - lower
    midpoint = (upper + lower) / 2.0

    sigma = width / (2 * z)

    z_obs = np.where(sigma > 0, (y_true - midpoint) / sigma, 0.0)

    phi_z = norm.pdf(z_obs)

    crps_values = sigma * (z_obs * (2 * norm.cdf(z_obs) - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))

    crps_values = np.where(sigma > 0, crps_values, np.abs(y_true - midpoint))

    return float(np.mean(crps_values))
