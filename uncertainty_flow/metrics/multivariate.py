"""Multivariate proper scoring rules: energy score and variogram score."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.distribution import DistributionPrediction


def _ensure_2d(y_arr: np.ndarray, d: int) -> np.ndarray:
    """Tile 1-D y_true across *d* columns for vectorised multivariate scoring."""
    if y_arr.ndim == 1:
        return np.tile(y_arr[:, None], (1, d))
    return y_arr


def energy_score(
    pred: DistributionPrediction,
    y_true,
    n_samples: int = 1000,
    random_state: int | None = None,
) -> float:
    """
    Energy score — a proper multivariate scoring rule.

    .. math::

        \\text{ES} = \\mathbb{E}[\\|X - y\\|] - \\frac{1}{2} \\mathbb{E}[\\|X - X'\\|]

    where :math:`X, X'` are independent draws from the forecast distribution.

    Args:
        pred: ``DistributionPrediction`` with ≥2 targets.
        y_true: True values (DataFrame / array matching targets).
        n_samples: Monte Carlo samples per observation.
        random_state: Random seed.

    Returns:
        Mean energy score (float). Lower is better.
    """
    y_arr = pred._coerce_y_true(y_true)
    n_obs = pred._n_samples
    d = len(pred._targets)

    if d < 2:
        raise ValueError("Energy score requires at least 2 targets.")

    rng = np.random.default_rng(random_state)
    rs1, rs2 = rng.integers(0, 2**31, size=2)

    sample_df1 = pred.sample(n_samples, random_state=int(rs1))
    sample_df2 = pred.sample(n_samples, random_state=int(rs2))

    targets = pred._targets
    sm1 = np.column_stack([sample_df1[t].to_numpy() for t in targets])
    sm1 = sm1.reshape(n_obs, n_samples, d)
    sm2 = np.column_stack([sample_df2[t].to_numpy() for t in targets])
    sm2 = sm2.reshape(n_obs, n_samples, d)

    y_arr = _ensure_2d(y_arr, d)

    term1 = np.zeros(n_obs)
    term2 = np.zeros(n_obs)

    for i in range(n_obs):
        samples_i = sm1[i]
        samples_i_prime = sm2[i]
        y_i = y_arr[i]

        diff_x_y = samples_i - y_i[None, :]
        norms_x_y = np.sqrt(np.sum(diff_x_y**2, axis=1))
        term1[i] = np.mean(norms_x_y)

        diff_xx = samples_i - samples_i_prime
        norms_xx = np.sqrt(np.sum(diff_xx**2, axis=1))
        term2[i] = 0.5 * np.mean(norms_xx)

    return float(np.mean(term1 - term2))


def variogram_score(
    pred: DistributionPrediction,
    y_true,
    n_samples: int = 1000,
    p: float = 0.5,
    random_state: int | None = None,
) -> float:
    """
    Variogram score — sensitive to correlation structure.

    .. math::

        \\text{VS}_p = \\sum_{i \\neq j} w_{ij}
            \\left(|y_i - y_j|^p - \\mathbb{E}[|X_i - X_j|^p]\\right)^2

    Args:
        pred: ``DistributionPrediction`` with ≥2 targets.
        y_true: True values.
        n_samples: Monte Carlo samples per observation.
        p: Power parameter (default 0.5).
        random_state: Random seed.

    Returns:
        Mean variogram score (float). Lower is better.
    """
    y_arr = pred._coerce_y_true(y_true)
    n_obs = pred._n_samples
    d = len(pred._targets)

    if d < 2:
        raise ValueError("Variogram score requires at least 2 targets.")

    sample_df = pred.sample(n_samples, random_state=random_state)

    targets = pred._targets
    sample_matrix = np.column_stack([sample_df[t].to_numpy() for t in targets])
    sample_matrix = sample_matrix.reshape(n_obs, n_samples, d)

    y_arr = _ensure_2d(y_arr, d)

    scores = np.zeros(n_obs)

    for i in range(n_obs):
        samples_i = sample_matrix[i]
        y_i = y_arr[i]

        vs = 0.0
        for j in range(d):
            for k in range(j + 1, d):
                obs_diff = np.abs(y_i[j] - y_i[k]) ** p
                sim_diffs = np.abs(samples_i[:, j] - samples_i[:, k]) ** p
                exp_diff = np.mean(sim_diffs)
                vs += (obs_diff - exp_diff) ** 2

        scores[i] = vs

    return float(np.mean(scores))
