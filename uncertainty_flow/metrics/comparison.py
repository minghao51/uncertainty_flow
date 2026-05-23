"""Model comparison tools for probabilistic forecasts.

Functions
---------
skill_score
    Relative improvement of one forecast over another.
diebold_mariano_test
    Diebold-Mariano test for predictive accuracy comparison.
model_confidence_set
    Hansen et al. (2011) model confidence set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import norm

if TYPE_CHECKING:
    from ..core.distribution import DistributionPrediction


def _extract_errors(
    pred: DistributionPrediction,
    y_true: np.ndarray,
    metric: str = "crps",
) -> np.ndarray:
    """Return per-sample errors for a given metric."""
    y_arr = pred._coerce_y_true(y_true)
    n = pred._n_samples
    n_targets = len(pred._targets)

    if y_arr.ndim == 1:
        y_by_target = y_arr[:, None]
    else:
        y_by_target = y_arr

    def _row_target_truth(i: int, t_idx: int) -> float:
        if y_by_target.shape[1] == 1:
            return float(y_by_target[i, 0])
        return float(y_by_target[i, t_idx])

    if metric in ("mae", "rmse"):
        median = pred.median()
        y_pred = median.to_numpy() if hasattr(median, "to_numpy") else np.asarray(median)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
        if metric == "mae":
            return np.mean(np.abs(y_by_target - y_pred), axis=1)
        return np.mean((y_by_target - y_pred) ** 2, axis=1)

    if metric == "crps":
        from .crps import crps_quantile

        scores = np.empty(n)

        for i in range(n):
            row_scores = []
            for t_idx in range(n_targets):
                q_start = t_idx * pred._n_quantiles
                q_end = q_start + pred._n_quantiles
                qv = pred._quantiles[i, q_start:q_end]
                row_scores.append(
                    crps_quantile(
                        np.array([_row_target_truth(i, t_idx)]),
                        qv[np.newaxis, :],
                        pred._levels,
                    )
                )
            scores[i] = float(np.mean(row_scores))

        return scores

    if metric == "pinball":
        from .pinball import pinball_loss

        scores = np.empty(n)

        for i in range(n):
            total = 0.0
            for level in pred._levels:
                for t_idx in range(len(pred._targets)):
                    q_start = t_idx * pred._n_quantiles
                    q_idx = pred._find_nearest_quantile_index(float(level))
                    q_val = pred._quantiles[i, q_start + q_idx]
                    y_val = _row_target_truth(i, t_idx)
                    total += pinball_loss(
                        np.array([y_val]),
                        np.array([q_val]),
                        float(level),
                    )
            scores[i] = total / (len(pred._levels) * len(pred._targets))

        return scores

    raise ValueError(f"Unsupported metric: {metric!r}")


def skill_score(
    pred_a: DistributionPrediction,
    pred_b: DistributionPrediction,
    y_true: np.ndarray,
    metric: str = "crps",
) -> pl.DataFrame:
    """Relative skill score of model A vs model B.

    The skill score (SS) is defined as:

        SS = 1 - score(A) / score(B)

    where SS > 0 means model A outperforms model B.

    Args:
        pred_a: Predictions from model A.
        pred_b: Predictions from model B (baseline).
        y_true: True values.
        metric: Scoring metric (``"crps"``, ``"mae"``, ``"rmse"``, ``"pinball"``).

    Returns:
        Polars DataFrame with columns ``metric``, ``score_a``, ``score_b``,
        ``skill_score``.
    """
    errors_a = _extract_errors(pred_a, y_true, metric=metric)
    errors_b = _extract_errors(pred_b, y_true, metric=metric)

    score_a = float(np.mean(errors_a))
    score_b = float(np.mean(errors_b))

    if score_b <= 0:
        ss = 0.0
    else:
        ss = 1.0 - score_a / score_b

    return pl.DataFrame(
        [
            {
                "metric": metric,
                "score_a": score_a,
                "score_b": score_b,
                "skill_score": ss,
            }
        ]
    )


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    one_sided: bool = True,
) -> pl.DataFrame:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests the null hypothesis that two sets of forecast errors have equal
    expected loss.

    Args:
        errors_a: Per-sample errors from model A.
        errors_b: Per-sample errors from model B.
        one_sided: If True (default), tests A < B (A has lower loss).
            If False, two-sided test for equal loss.

    Returns:
        Polars DataFrame with columns ``dm_statistic``, ``p_value``,
        ``result`` (reject or not), ``better_model``.
    """
    errors_a = np.asarray(errors_a, dtype=float).ravel()
    errors_b = np.asarray(errors_b, dtype=float).ravel()

    if len(errors_a) != len(errors_b):
        raise ValueError(
            f"error arrays must have same length, got {len(errors_a)} vs {len(errors_b)}"
        )

    dm_stat, p_value = _dm_statistic(errors_a, errors_b, one_sided=one_sided)

    alpha = 0.05
    reject = p_value < alpha

    if reject:
        better = "A" if float(np.mean(errors_a - errors_b)) < 0 else "B"
    else:
        better = "tie"

    return pl.DataFrame(
        [
            {
                "dm_statistic": dm_stat,
                "p_value": p_value,
                "result": "reject" if reject else "not_reject",
                "better_model": better,
            }
        ]
    )


def _dm_statistic(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    one_sided: bool = True,
) -> tuple[float, float]:
    """Compute DM test statistic and p-value."""
    d = np.asarray(errors_a, dtype=float).ravel() - np.asarray(errors_b, dtype=float).ravel()
    n = len(d)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))

    if var_d <= 0 or n < 2:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(var_d / n)
    if one_sided:
        p_value = float(norm.cdf(dm_stat))
    else:
        p_value = 2.0 * (1.0 - float(norm.cdf(abs(dm_stat))))

    return dm_stat, p_value


def model_confidence_set(
    predictions: dict[str, DistributionPrediction],
    y_true: np.ndarray,
    metric: str = "crps",
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Hansen et al. (2011) Model Confidence Set.

    Sequentially eliminates models that are significantly inferior to the
    best-performing model. Uses the Diebold-Mariano test for pairwise
    comparisons with a Bonferroni correction.

    Args:
        predictions: Dict mapping model names to ``DistributionPrediction``
            objects.
        y_true: True values.
        metric: Scoring metric (``"crps"``, ``"mae"``, ``"rmse"``, ``"pinball"``).
        alpha: Significance level (default 0.05).

    Returns:
        Polars DataFrame with columns ``model``, ``score``, ``in_set``
        (whether the model survives in the confidence set).
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)

    if n_models < 2:
        err = _extract_errors(predictions[model_names[0]], y_true, metric)
        results = [
            {
                "model": model_names[0],
                "score": float(np.mean(err)),
                "in_set": True,
            }
        ]
        return pl.DataFrame(results)

    errors: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for name in model_names:
        err = _extract_errors(predictions[name], y_true, metric)
        errors[name] = err
        scores[name] = float(np.mean(err))

    surviving = set(model_names)

    changed = True
    while changed and len(surviving) > 1:
        changed = False
        eliminations: set[str] = set()

        surv_list = sorted(surviving)
        n_pairs = len(surv_list) * (len(surv_list) - 1) // 2
        corrected_alpha = alpha / max(1, n_pairs)

        for i in range(len(surv_list)):
            for j in range(i + 1, len(surv_list)):
                a, b = surv_list[i], surv_list[j]

                _, p_value = _dm_statistic(errors[a], errors[b], one_sided=False)

                if p_value < corrected_alpha:
                    mean_d = float(np.mean(errors[a] - errors[b]))
                    if mean_d > 0:
                        eliminations.add(a)
                    else:
                        eliminations.add(b)
                    changed = True

        surviving -= eliminations

    return pl.DataFrame(
        [
            {
                "model": name,
                "score": scores[name],
                "in_set": name in surviving,
            }
            for name in model_names
        ]
    )
