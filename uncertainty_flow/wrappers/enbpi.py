"""Ensemble Bootstrap Prediction Intervals (EnbPI).

Implements Xu & Xie (2021): trains an ensemble of bootstrap base learners
and constructs prediction intervals calibrated via sequential conformal
nonconformity scores. Designed for time series with distribution shift.

The ensemble spread provides a flexible uncertainty estimate, and the
conformal calibration ensures marginal coverage guarantees. Supports
sequential score updates after each observation (similar to ACI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, clone

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput, TargetSpec
from ..utils.exceptions import ConfigurationError, error_model_not_fitted
from ..utils.polars_bridge import materialize_lazyframe, to_numpy

if TYPE_CHECKING:
    pass


class EnsembleBootstrapPI(BaseUncertaintyModel):
    """Ensemble Bootstrap Prediction Intervals (EnbPI).

    Trains ``n_estimators`` bootstrap copies of a sklearn regressor, then
    constructs prediction intervals from the ensemble distribution with
    conformal calibration via stored nonconformity scores.

    Usage pattern:
        1. ``fit(train_data, target)`` — trains the bootstrap ensemble.
        2. ``predict(test_data)`` — returns calibrated prediction intervals.
        3. ``update(y_true)`` — after observing the true value, updates the
           nonconformity score pool for future predictions.

    Examples:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> import polars as pl
        >>> from uncertainty_flow.wrappers import EnsembleBootstrapPI
        >>>
        >>> df = pl.DataFrame({
        ...     "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     "y": [1.5, 3.5, 5.5, 7.5, 9.5, 9.0, 7.0, 5.0, 3.0, 1.0],
        ... })
        >>> model = EnsembleBootstrapPI(
        ...     base_model=GradientBoostingRegressor(random_state=42),
        ...     n_estimators=20,
        ...     random_state=42,
        ... )
        >>> model.fit(df, target="y")
        >>> pred = model.predict(df)
        >>> model.update(df["y"][0])
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        n_estimators: int = 100,
        coverage_target: float = 0.9,
        subsample_ratio: float = 1.0,
        random_state: int | None = None,
    ):
        if n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2, got {n_estimators}")
        if not (0 < coverage_target < 1):
            raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}")
        if not (0 < subsample_ratio <= 1):
            raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio}")

        self.base_model = base_model
        self.n_estimators = n_estimators
        self.coverage_target = coverage_target
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

        self._fitted = False
        self._models: list[BaseEstimator] = []
        self._feature_cols_: list[str] = []
        self._target_col_: str = ""
        self._scores: list[float] = []
        self._quantile_levels_: np.ndarray | None = None
        self._last_preds: np.ndarray | None = None

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> EnsembleBootstrapPI:
        data = materialize_lazyframe(data)

        if target is None:
            raise ConfigurationError("target is required for EnsembleBootstrapPI")
        target_str = target if isinstance(target, str) else target[0]
        self._target_col_ = target_str

        if target_str not in data.columns:
            raise ValueError(
                f"Target column '{target_str}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        self._feature_cols_ = [c for c in data.columns if c != target_str]
        if not self._feature_cols_:
            raise ValueError("No feature columns remaining after excluding target.")

        x_all = to_numpy(data, self._feature_cols_)
        y_all = data[target_str].to_numpy().flatten()
        n = len(data)
        subsample_size = max(1, int(n * self.subsample_ratio))

        rng = np.random.default_rng(self.random_state)
        self._models = []

        for i in range(self.n_estimators):
            boot_idx = rng.integers(0, n, size=subsample_size)
            x_boot = x_all[boot_idx]
            y_boot = y_all[boot_idx]

            model = clone(self.base_model)
            seed = (self.random_state if self.random_state is not None else 42) + i
            if "random_state" in model.get_params(deep=False):
                model.set_params(random_state=seed)
            model.fit(x_boot, y_boot)
            self._models.append(model)

        calib_preds = np.mean([m.predict(x_all) for m in self._models], axis=0)
        residuals = np.abs(y_all - calib_preds)
        self._scores = residuals.tolist()
        self._quantile_levels_ = np.asarray(list(DEFAULT_QUANTILES), dtype=float)

        self._fitted = True
        return self

    def predict(
        self,
        data: PolarsInput,
    ) -> DistributionPrediction:
        if not self._fitted:
            error_model_not_fitted("EnsembleBootstrapPI")

        data = materialize_lazyframe(data)
        x = to_numpy(data, self._feature_cols_)

        all_preds = np.column_stack([m.predict(x) for m in self._models])
        point_preds = np.mean(all_preds, axis=1)

        # Store the last point predictions for update() to compute residuals
        self._last_preds = point_preds.copy()

        score_arr = np.array(self._scores) if self._scores else np.array([0.0])
        conformal_level = min(self.coverage_target, 1.0)
        q_value = np.quantile(score_arr, conformal_level)

        if self._quantile_levels_ is None:
            raise RuntimeError("Internal error: quantile levels not set")

        quantile_matrix = np.zeros((len(point_preds), len(self._quantile_levels_)))
        ens_std = np.std(all_preds, axis=1)
        ens_std = np.clip(ens_std, 1e-12, None)

        for j, level in enumerate(self._quantile_levels_):
            z = float(
                np.percentile(
                    (all_preds - point_preds[:, None]).ravel()
                    / np.repeat(ens_std, self.n_estimators),
                    level * 100,
                )
            )
            quantile_matrix[:, j] = point_preds + z * ens_std + q_value * (level - 0.5)

        quantile_matrix = np.sort(quantile_matrix, axis=1)

        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=self._quantile_levels_.tolist(),
            target_names=[self._target_col_],
        )

    def update(self, y_true: float | np.ndarray) -> None:
        if not self._fitted:
            error_model_not_fitted("EnsembleBootstrapPI")

        y_arr = np.atleast_1d(np.asarray(y_true, dtype=float))

        if self._last_preds is None:
            raise RuntimeError(
                "update() called before predict(). "
                "Call predict() first to generate predictions, "
                "then update() with the corresponding true values."
            )

        if len(y_arr) != len(self._last_preds):
            raise ValueError(
                f"y_true length ({len(y_arr)}) must match the number of "
                f"predictions from the most recent predict() call ({len(self._last_preds)})."
            )

        for i in range(len(y_arr)):
            residual = abs(float(y_arr[i]) - float(self._last_preds[i]))
            self._scores.append(residual)
