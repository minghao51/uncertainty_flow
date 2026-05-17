"""Log-score (negative log-likelihood) for probabilistic predictions."""

from __future__ import annotations

import warnings

import numpy as np


def _coerce_inputs(
    y_true: np.ndarray,
    quantile_matrix: np.ndarray,
    quantile_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(quantile_matrix, dtype=np.float64),
        np.asarray(quantile_levels, dtype=np.float64),
    )


def _mean_finite_log(log_densities: np.ndarray) -> float:
    finite_mask = np.isfinite(log_densities)
    return float(np.mean(log_densities[finite_mask])) if np.any(finite_mask) else -np.inf


def log_score(
    y_true: np.ndarray,
    quantile_matrix: np.ndarray,
    quantile_levels: np.ndarray,
    family: str = "auto",
) -> float:
    """
    Mean log-score from quantile predictions via parametric fitting.

    Fits one parametric distribution per observation (row), then evaluates
    the log-density at that row's true value.

    This preserves row-wise uncertainty structure (e.g. heteroscedasticity).
    For the older pooled behavior, use :func:`log_score_pooled`.

    For a non-parametric fallback, see :func:`log_score_kde`.

    Args:
        y_true: (n,) array of true values.
        quantile_matrix: (n, k) array of predicted quantile values.
        quantile_levels: (k,) array of quantile levels.
        family: Parametric family or ``"auto"``.

    Returns:
        Mean log-score (float). Higher is better.
    """
    y_true, quantile_matrix, quantile_levels = _coerce_inputs(
        y_true, quantile_matrix, quantile_levels
    )

    from ..core.parametric import fit_parametric

    if quantile_matrix.ndim != 2:
        raise ValueError("quantile_matrix must be 2-D with shape (n_samples, n_quantiles)")
    if len(y_true) != quantile_matrix.shape[0]:
        raise ValueError(
            f"y_true length ({len(y_true)}) must match quantile_matrix rows "
            f"({quantile_matrix.shape[0]})"
        )

    log_densities = np.empty(len(y_true), dtype=np.float64)
    for i in range(len(y_true)):
        dist = fit_parametric(quantile_matrix[i], quantile_levels, family=family)
        log_densities[i] = float(dist.logpdf(y_true[i]))

    finite_mask = np.isfinite(log_densities)
    if not np.all(finite_mask):
        warnings.warn(
            f"{int(np.sum(~finite_mask))} log-density values are non-finite. "
            "Check that y_true values are in the support of the fitted distribution.",
            UserWarning,
            stacklevel=2,
        )

    return _mean_finite_log(log_densities)


def log_score_pooled(
    y_true: np.ndarray,
    quantile_matrix: np.ndarray,
    quantile_levels: np.ndarray,
    family: str = "auto",
) -> float:
    """
    Mean log-score using one pooled distribution fitted from mean quantiles.

    This helper keeps the previous behavior for backward comparisons.
    """
    y_true, quantile_matrix, quantile_levels = _coerce_inputs(
        y_true, quantile_matrix, quantile_levels
    )

    from ..core.parametric import fit_parametric

    mean_qv = np.mean(quantile_matrix, axis=0)
    dist = fit_parametric(mean_qv, quantile_levels, family=family)
    log_densities = dist.logpdf(y_true)

    return _mean_finite_log(log_densities)


def log_score_kde(
    y_true: np.ndarray,
    quantile_matrix: np.ndarray,
    quantile_levels: np.ndarray,
    n_draw: int = 500,
    random_state: int | None = None,
) -> float:
    """
    Non-parametric log-score via kernel density estimation.

    Draws samples from the piecewise-linear CDF defined by the quantile
    knots, fits a Gaussian KDE, and evaluates the log-density.

    Args:
        y_true: (n,) array of true values.
        quantile_matrix: (n, k) array of predicted quantile values.
        quantile_levels: (k,) array of quantile levels.
        n_draw: Samples per observation for KDE fitting.
        random_state: Random seed.

    Returns:
        Mean log-score (float).
    """
    from scipy.stats import gaussian_kde

    y_true, quantile_matrix, quantile_levels = _coerce_inputs(
        y_true, quantile_matrix, quantile_levels
    )

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    log_densities = np.empty(n)

    for i in range(n):
        qv = quantile_matrix[i]
        u = rng.uniform(0, 1, size=n_draw)
        u_clipped = np.clip(u, quantile_levels[0], quantile_levels[-1])
        samples = np.interp(u_clipped, quantile_levels, qv)

        try:
            kde = gaussian_kde(samples)
            log_densities[i] = float(kde.logpdf(y_true[i])[0])
        except (np.linalg.LinAlgError, ValueError):
            log_densities[i] = -np.inf

    return _mean_finite_log(log_densities)
