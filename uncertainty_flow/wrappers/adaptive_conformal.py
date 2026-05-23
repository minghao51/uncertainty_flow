"""Adaptive Conformal Inference (ACI) for time series.

Implements Gibbs & Candes (2021): maintains a running error budget alpha_t
that adjusts after each observation. Under distribution shift, coverage
degrades for static conformal methods; ACI adapts by widening or narrowing
intervals based on recent coverage performance.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import PolarsInput, TargetSpec
from ..utils.exceptions import ModelNotFittedError
from ..utils.polars_bridge import materialize_lazyframe, to_numpy_series


class AdaptiveConformalForecaster(BaseUncertaintyModel):
    """
    Adaptive Conformal Inference wrapper for sequential prediction.

    Wraps a fitted ``BaseUncertaintyModel`` and adjusts prediction interval
    width dynamically. After each observation, call :meth:`update` (or
    :meth:`update_batch`) to adapt the coverage level.

    The adaptive rule (Gibbs & Candes 2021):

        alpha_{t+1} = alpha_t + gamma * (alpha - 1(|y_t - yhat_t| > q_{1-alpha_t}))

    where ``alpha`` is the *fixed* target miscoverage level. If recent
    coverage is too low (errors exceed intervals), alpha_t decreases,
    widening intervals. If coverage is too high, alpha_t increases,
    narrowing intervals.

    Examples:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from uncertainty_flow.wrappers import ConformalRegressor
        >>> from uncertainty_flow.wrappers import AdaptiveConformalForecaster
        >>>
        >>> base = ConformalRegressor(GradientBoostingRegressor())
        >>> base.fit(df_train, target="y")
        >>> aci = AdaptiveConformalForecaster(model=base)
        >>> aci.fit(df_calib, target="y")
        >>>
        >>> for t in range(n_steps):
        ...     pred = aci.predict(df.iloc[t:t+1])
        ...     aci.update(y_true[t])
    """

    def __init__(
        self,
        model: BaseUncertaintyModel,
        alpha: float = 0.1,
        gamma: float = 0.01,
    ):
        """
        Args:
            model: A fitted BaseUncertaintyModel.
            alpha: Initial miscoverage level (default 0.1 → 90% coverage).
            gamma: Learning rate for alpha adaptation (default 0.01).
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.model = model
        self._initial_alpha = alpha
        self.gamma = gamma

        self._fitted = False
        self._alpha_t = alpha
        self._scores: list[float] = []
        self._feature_cols: list[str] = []
        self._target_col: str = ""
        # Stored from the last predict() call for use in update()
        self._last_point_pred: np.ndarray | None = None
        self._last_q_value: float | None = None
        self._last_n_targets: int = 1

    @property
    def current_alpha(self) -> float:
        """Current adaptive miscoverage level."""
        return self._alpha_t

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> AdaptiveConformalForecaster:
        """
        Initialize ACI with calibration conformal scores.

        Args:
            data: Calibration dataset for computing initial nonconformity scores.
            target: Target column name.

        Returns:
            self
        """
        data = materialize_lazyframe(data)

        if target is None:
            from ..utils.exceptions import ConfigurationError

            raise ConfigurationError("target is required for AdaptiveConformalForecaster")
        target_str = target if isinstance(target, str) else target[0]
        self._target_col = target_str

        self._feature_cols = [c for c in data.columns if c != target_str]

        pred = self.model.predict(data)
        y_true = to_numpy_series(data[target_str])

        if len(pred._targets) == 1:
            median_vals = pred.median()
            if isinstance(median_vals, pl.DataFrame):
                point_preds = median_vals.to_numpy().ravel()
            else:
                point_preds = median_vals.to_numpy()
        else:
            median_result = pred.median()
            if isinstance(median_result, pl.DataFrame):
                point_preds = median_result[target_str].to_numpy()
            else:
                point_preds = median_result.to_numpy()

        residuals = np.abs(y_true - point_preds)
        self._scores = residuals.tolist()
        self._alpha_t = self._initial_alpha
        self._fitted = True
        return self

    def predict(
        self,
        data: PolarsInput,
        steps: int = 1,
    ) -> DistributionPrediction:
        """
        Generate adaptive prediction intervals.

        Args:
            data: Input data for prediction.
            steps: Number of steps ahead (propagates alpha adjustment for
                multi-step forecasts). Default 1.

        Returns:
            DistributionPrediction with intervals reflecting current alpha_t.
        """
        if not self._fitted:
            raise ModelNotFittedError("AdaptiveConformalForecaster")

        data = materialize_lazyframe(data)
        pred = self.model.predict(data)

        n_pred = pred._n_samples
        n_quantiles = pred._n_quantiles

        alphas = self._propagate_alpha(steps)
        alpha = alphas[-1]

        score_arr = np.array(self._scores)
        if len(score_arr) > 0:
            q_value = np.quantile(score_arr, min(1 - alpha, 1.0))
        else:
            q_value = 0.0

        # Scale factors: map quantile levels to conformal interval bounds.
        # The outermost levels should map to +/- q_value around the median.
        lower_scale = max(0.5 - pred._levels[0], 1e-12)
        upper_scale = max(pred._levels[-1] - 0.5, 1e-12)

        def _apply_conformal_scaling(point_preds: np.ndarray, n_quantiles: int) -> np.ndarray:
            result = np.empty((len(point_preds), n_quantiles))
            for j in range(n_quantiles):
                level = pred._levels[j]
                if level < 0.5:
                    result[:, j] = point_preds - q_value * (0.5 - level) / lower_scale
                elif level > 0.5:
                    result[:, j] = point_preds + q_value * (level - 0.5) / upper_scale
                else:
                    result[:, j] = point_preds
            return result

        if len(pred._targets) == 1:
            median_idx = pred._find_nearest_quantile_index(0.5)
            point_preds = pred._quantiles[:, median_idx].copy()
            output_quantiles = _apply_conformal_scaling(point_preds, n_quantiles)
        else:
            output_quantiles = pred._quantiles.copy()
            for t_idx in range(len(pred._targets)):
                q_start = t_idx * n_quantiles
                median_idx = pred._find_nearest_quantile_index(0.5)
                point_preds = pred._quantiles[:, q_start + median_idx].copy()
                scaled = _apply_conformal_scaling(point_preds, n_quantiles)
                output_quantiles[:, q_start : q_start + n_quantiles] = scaled

        output_quantiles = np.sort(output_quantiles, axis=1)

        # Store the last prediction's point estimates and conformal quantile
        # for use in the subsequent update() call.
        if n_pred > 0:
            n_targets = len(pred._targets)
            self._last_n_targets = n_targets
            if n_targets == 1:
                self._last_point_pred = np.array([float(point_preds[-1])])
            else:
                last_point_preds = np.empty(n_targets)
                for t_idx in range(n_targets):
                    q_start = t_idx * n_quantiles
                    median_idx_t = pred._find_nearest_quantile_index(0.5)
                    last_point_preds[t_idx] = float(output_quantiles[-1, q_start + median_idx_t])
                self._last_point_pred = last_point_preds
            self._last_q_value = float(q_value)

        return DistributionPrediction(
            quantile_matrix=output_quantiles,
            quantile_levels=pred._levels.tolist(),
            target_names=list(pred._targets),
        )

    def update(self, y_true: float | int | np.ndarray) -> None:
        """
        Update alpha after observing a true value.

        Must be called after :meth:`predict` with the corresponding true
        observation. Updates internal conformal scores and adapts alpha.

        For univariate models, pass a scalar. For multivariate, pass an
        array-like with one value per target.

        Args:
            y_true: Observed true value (scalar for univariate, array for
                multivariate).
        """
        if not self._fitted:
            raise ModelNotFittedError("AdaptiveConformalForecaster")

        if self._last_point_pred is None:
            raise RuntimeError(
                "update() called before predict(). "
                "Call predict() first to generate a prediction for this time step."
            )

        y_arr = np.atleast_1d(np.asarray(y_true, dtype=float))

        if y_arr.shape[0] != self._last_n_targets:
            raise ValueError(
                f"y_true has {y_arr.shape[0]} value(s) but model has "
                f"{self._last_n_targets} target(s)."
            )

        if self._last_n_targets == 1:
            new_score = abs(float(y_arr[0]) - self._last_point_pred[0])
        else:
            new_score = float(np.mean(np.abs(y_arr - self._last_point_pred)))

        q_value = self._last_q_value if self._last_q_value is not None else 0.0
        exceeded = 1.0 if new_score > q_value else 0.0

        self._alpha_t = self._alpha_t + self.gamma * (self._initial_alpha - exceeded)
        self._alpha_t = max(1e-6, min(self._alpha_t, 1.0 - 1e-6))

        self._scores.append(new_score)
        max_scores = 1000
        if len(self._scores) > max_scores:
            self._scores = self._scores[-max_scores:]

    def update_batch(self, y_true: pl.Series | np.ndarray) -> None:
        """
        Sequentially update alpha for a batch of observations.

        Calls :meth:`update` for each observation in order. This is
        equivalent to calling ``update()`` in a loop but more convenient
        for offline replay and testing.

        .. note::
            This method assumes you have already called ``predict()`` for
            each corresponding observation and are now providing the true
            values in sequence. It does NOT call ``predict()`` internally.

        Args:
            y_true: Array or Series of observed true values.
                Shape ``(n,)`` for univariate, ``(n, n_targets)`` for
                multivariate.
        """
        if isinstance(y_true, pl.Series):
            y_arr = to_numpy_series(y_true)
        else:
            y_arr = np.asarray(y_true, dtype=float)

        if y_arr.ndim == 1:
            for y in y_arr:
                self.update(float(y))
        else:
            for row in y_arr:
                self.update(row)

    def _propagate_alpha(self, steps: int) -> list[float]:
        """Project alpha forward for multi-step prediction (no actual update).

        Without observing future values, alpha cannot be adapted. This method
        returns a constant projection (current alpha repeated), which is a
        conservative default.
        """
        return [self._alpha_t] * steps
