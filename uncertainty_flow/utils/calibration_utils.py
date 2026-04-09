"""Calibration utility functions for uncertainty models."""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ..core.base import BaseUncertaintyModel

from ..metrics import coverage_score, winkler_score
from ..utils.exceptions import warn_coverage_gap
from .polars_bridge import to_numpy_series_zero_copy


def calibration_report(
    model: "BaseUncertaintyModel",
    data: pl.DataFrame,
    target: str | list[str],
    quantile_levels: list[float] | None = None,
) -> pl.DataFrame:
    """
    Generate calibration report for a fitted model.

    Args:
        model: Fitted uncertainty model
        data: Validation data
        target: Target column name(s)
        quantile_levels: Quantile levels to evaluate (default: [0.8, 0.9, 0.95])

    Returns:
        Polars DataFrame with calibration metrics:
        - quantile: Quantile level
        - requested_coverage: Requested coverage
        - achieved_coverage: Actual empirical coverage
        - sharpness: Average interval width
        - winkler_score: Winkler interval score

    Examples:
        >>> from uncertainty_flow.wrappers import ConformalRegressor
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> model = ConformalRegressor(GradientBoostingRegressor())
        >>> model.fit(train_df, target="price")
        >>> report = model.calibration_report(val_df, target="price")
        >>> print(report)
    """
    if quantile_levels is None:
        quantile_levels = [0.8, 0.9, 0.95]

    # Handle multivariate targets
    if isinstance(target, str):
        targets = [target]
    else:
        targets = target

    results = []

    for level in quantile_levels:
        pred = model.predict(data)
        intervals = pred.interval(confidence=level)

        # Get actuals and predictions
        row_results: dict[str, float] = {}
        sharpness_sum: float = 0.0
        winkler_sum: float = 0.0
        total_coverage: float = 0.0
        n_targets: int = 0

        for t in targets:
            actuals = data[t]
            if len(targets) == 1:
                lower = intervals["lower"]
                upper = intervals["upper"]
            else:
                lower = intervals[f"{t}_lower"]
                upper = intervals[f"{t}_upper"]

            achieved = coverage_score(actuals, lower, upper)
            sharpness_values = to_numpy_series_zero_copy(upper - lower)
            sharpness = float(np.mean(sharpness_values))
            winkler = winkler_score(actuals, lower, upper, level)

            row_results[f"{t}_achieved"] = achieved
            sharpness_sum += sharpness
            winkler_sum += winkler
            total_coverage += achieved
            n_targets += 1

        # Average across targets
        avg_coverage = total_coverage / n_targets
        avg_sharpness = sharpness_sum / n_targets
        avg_winkler = winkler_sum / n_targets

        results.append(
            {
                "quantile": level,
                "requested_coverage": level,
                "achieved_coverage": avg_coverage,
                "sharpness": avg_sharpness,
                "winkler_score": avg_winkler,
            }
        )

        # Warn if coverage gap > 5%
        if abs(avg_coverage - level) > 0.05:
            warn_coverage_gap(level, avg_coverage)

    return pl.DataFrame(results)
