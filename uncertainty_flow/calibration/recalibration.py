"""Isotonic recalibration wrapper for probabilistic predictions.

Implements the method of Kuleshov et al. (2018): fit isotonic regression
mapping predicted quantile levels to empirical coverage on a calibration set,
then remap predictions at inference time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import PolarsInput, TargetSpec
from ..utils.exceptions import error_model_not_fitted
from ..utils.polars_bridge import materialize_lazyframe, to_numpy_series_zero_copy

if TYPE_CHECKING:
    pass


class RecalibratedModel(BaseUncertaintyModel):
    """
    Wrap a fitted model with isotonic recalibration.

    Learns a monotone mapping from predicted quantile levels to empirical
    coverage using isotonic regression (Kuleshov et al., 2018).

    Supports two modes:
    - **Separate calibration set** (default): fit the isotonic map on held-out
      data provided to ``fit()``.
    - **Cross-fitting** (``cross_calibrate=True``): K-fold fit on the inner
      model's training data to avoid overfitting the isotonic map.

    Examples:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from uncertainty_flow.wrappers import ConformalRegressor
        >>> from uncertainty_flow.calibration import RecalibratedModel
        >>> import polars as pl
        >>>
        >>> base = ConformalRegressor(base_model=GradientBoostingRegressor())
        >>> base.fit(df_train, target="y")
        >>> recal = RecalibratedModel(model=base)
        >>> recal.fit(df_calib, target="y")
        >>> pred = recal.predict(df_test)
    """

    def __init__(
        self,
        model: BaseUncertaintyModel,
        quantile_levels: list[float] | None = None,
        cross_calibrate: bool = False,
        n_folds: int = 5,
        random_state: int | None = None,
    ):
        """
        Args:
            model: A fitted BaseUncertaintyModel to recalibrate.
            quantile_levels: Quantile levels for the output predictions.
                Defaults to the inner model's levels.
            cross_calibrate: If True, use K-fold cross-fitting on the
                calibration data to avoid overfitting the isotonic map.
            n_folds: Number of folds when cross_calibrate=True.
            random_state: Random seed for cross-fitting.
        """
        self.model = model
        self.quantile_levels = quantile_levels
        self.cross_calibrate = cross_calibrate
        self.n_folds = n_folds
        self.random_state = random_state

        self._fitted = False
        self._isotonic_regressors: list[IsotonicRegression] | None = None
        self._output_levels: np.ndarray | None = None
        self._target_names: list[str] | None = None

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> RecalibratedModel:
        """
        Learn the isotonic recalibration mapping.

        Args:
            data: Calibration dataset. If cross_calibrate=True, this data
                is split into K folds for cross-fitting.
            target: Target column name(s).

        Returns:
            self
        """
        data = materialize_lazyframe(data)

        if self.cross_calibrate:
            self._fit_cross_calibrated(data, target)
        else:
            self._fit_direct(data, target)

        self._fitted = True
        return self

    def _fit_direct(
        self,
        data: pl.DataFrame,
        target: TargetSpec | None,
    ) -> None:
        pred = self.model.predict(data)
        y_arr = self._extract_y(data, target, pred)

        self._target_names = pred._targets
        self._output_levels = (
            np.asarray(self.quantile_levels, dtype=float)
            if self.quantile_levels is not None
            else pred._levels.copy()
        )

        self._isotonic_regressors = self._fit_isotonic_map(pred, y_arr)

    def _fit_cross_calibrated(
        self,
        data: pl.DataFrame,
        target: TargetSpec | None,
    ) -> None:
        from sklearn.model_selection import KFold

        n = len(data)
        n_splits = min(self.n_folds, n)
        if n_splits < 2:
            # Fall back to direct fit if too few samples for cross-fitting
            self._fit_direct(data, target)
            return

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        ref_pred = self.model.predict(data)
        self._target_names = ref_pred._targets
        self._output_levels = (
            np.asarray(self.quantile_levels, dtype=float)
            if self.quantile_levels is not None
            else ref_pred._levels.copy()
        )

        n_targets = len(ref_pred._targets)
        n_quantiles = ref_pred._n_quantiles

        # Accumulate empirical coverage per fold per quantile level
        fold_empirical_sums = np.zeros((n_targets, n_quantiles))
        fold_counts = 0

        for train_idx, val_idx in kf.split(range(n)):
            val_df = (
                data.with_row_index("__idx__")
                .filter(pl.col("__idx__").is_in(list(val_idx)))
                .drop("__idx__")
            )

            fold_pred = self.model.predict(val_df)
            y_val = self._extract_y(val_df, target, fold_pred)

            for t_idx in range(n_targets):
                q_start = t_idx * fold_pred._n_quantiles
                q_end = q_start + fold_pred._n_quantiles
                q_slice = fold_pred._quantiles[:, q_start:q_end]
                y_col = y_val[:, t_idx] if y_val.ndim == 2 else y_val

                empirical = np.mean(y_col[:, None] <= q_slice, axis=0)
                fold_empirical_sums[t_idx] += empirical

            fold_counts += 1

        avg_empirical = fold_empirical_sums / max(fold_counts, 1)

        self._isotonic_regressors = []
        for t_idx in range(n_targets):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(ref_pred._levels, avg_empirical[t_idx])
            self._isotonic_regressors.append(iso)

    def predict(self, data: PolarsInput) -> DistributionPrediction:
        """
        Generate recalibrated predictions.

        Args:
            data: Input data for prediction.

        Returns:
            DistributionPrediction with recalibrated quantile values.
        """
        if not self._fitted:
            error_model_not_fitted("RecalibratedModel")

        data = materialize_lazyframe(data)
        pred = self.model.predict(data)

        if self._isotonic_regressors is None:
            raise RuntimeError("Internal error: isotonic regressors not fitted")
        if self._output_levels is None:
            raise RuntimeError("Internal error: output levels not set")
        if self._target_names is None:
            raise RuntimeError("Internal error: target names not set")

        n_targets = len(pred._targets)
        n_quantiles = pred._n_quantiles
        n_out = len(self._output_levels)

        output_matrix = np.empty((pred._n_samples, n_targets * n_out))

        for t_idx in range(n_targets):
            q_start = t_idx * n_quantiles
            q_end = q_start + n_quantiles
            q_slice = pred._quantiles[:, q_start:q_end]

            iso = self._isotonic_regressors[t_idx]

            # Build inverse mapping: for desired empirical coverage,
            # find the nominal quantile level that produces it.
            # iso maps nominal_level -> empirical_coverage.
            # We evaluate on the model's levels (which are increasing) to get
            # empirical coverage at each level, then interpolate back.
            empirical_at_levels = iso.predict(pred._levels)

            out_q = np.empty((pred._n_samples, n_out))
            for j in range(n_out):
                level = self._output_levels[j]

                # Find nominal level such that empirical coverage ≈ level
                # Since both empirical_at_levels and pred._levels are
                # monotone increasing, np.interp gives the inverse.
                if empirical_at_levels[-1] <= empirical_at_levels[0]:
                    # Degenerate isotonic map (should not happen)
                    input_level = level
                else:
                    input_level = float(np.interp(level, empirical_at_levels, pred._levels))

                # Find nearest model quantile to input_level
                nearest_idx = int(np.argmin(np.abs(pred._levels - input_level)))
                out_q[:, j] = q_slice[:, nearest_idx]

            o_start = t_idx * n_out
            o_end = o_start + n_out
            output_matrix[:, o_start:o_end] = out_q

        output_matrix = np.sort(output_matrix, axis=1)

        return DistributionPrediction(
            quantile_matrix=output_matrix,
            quantile_levels=self._output_levels.tolist(),
            target_names=list(self._target_names),
        )

    def _fit_isotonic_map(
        self,
        pred: DistributionPrediction,
        y_arr: np.ndarray,
    ) -> list[IsotonicRegression]:
        n_targets = len(pred._targets)
        regressors: list[IsotonicRegression] = []

        for t_idx in range(n_targets):
            q_start = t_idx * pred._n_quantiles
            q_end = q_start + pred._n_quantiles
            q_slice = pred._quantiles[:, q_start:q_end]
            y_col = y_arr[:, t_idx] if y_arr.ndim == 2 else y_arr

            # Empirical coverage at each nominal quantile level:
            # fraction of observations where true value <= predicted quantile
            empirical_coverage = np.mean(y_col[:, None] <= q_slice, axis=0)

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pred._levels, empirical_coverage)
            regressors.append(iso)

        return regressors

    @staticmethod
    def _extract_y(
        data: pl.DataFrame,
        target: TargetSpec | None,
        pred: DistributionPrediction,
    ) -> np.ndarray:
        targets = pred._targets
        cols = [to_numpy_series_zero_copy(data[t]) for t in targets]
        if len(cols) == 1:
            return cols[0]
        return np.column_stack(cols)
