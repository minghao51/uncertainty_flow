"""ConformalForecaster - time series forecasting with conformal prediction."""

import warnings

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput, TargetSpec
from ..multivariate.copula import COPULA_FAMILIES, BaseCopula, auto_select_copula
from ..utils.auto_tuning import (
    build_tune_splits,
    candidate_values,
    estimator_param_candidates,
    score_distribution_prediction,
    valid_calibration_candidates,
)
from ..utils.exceptions import InvalidDataError, ModelNotFittedError
from ..utils.polars_bridge import materialize_lazyframe, to_numpy
from ..utils.split import select_validation_plan


class ConformalForecaster(BaseUncertaintyModel):
    """
    Time series forecasting with conformal prediction.

    Coverage guarantee: ✅ (with temporal correction)
    Non-crossing: ✅ (post-sort)

    Examples:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from uncertainty_flow.wrappers import ConformalForecaster
        >>> import polars as pl
        >>>
        >>> df = pl.DataFrame({
        ...     "date": range(10),
        ...     "price": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ... })
        >>> model = ConformalForecaster(
        ...     base_model=GradientBoostingRegressor(),
        ...     targets="price",
        ...     horizon=3,
        ...     lags=2,
        ... )
        >>> model.fit(df)
        >>> pred = model.predict(df)
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        horizon: int,
        targets: str | list[str],
        copula_family: str = "auto",
        lags: int | list[int] = 1,
        calibration_method: str = "holdout",
        calibration_size: float = 0.2,
        auto_tune: bool = True,
        uncertainty_features: list[str] | None = None,
        random_state: int | None = None,
    ):
        """
        Initialize ConformalForecaster.

        Args:
            base_model: Any sklearn-compatible regressor
            horizon: Forecast horizon (steps ahead)
            targets: Target column name(s)
            copula_family: (
                "auto" (BIC selection) or one of "gaussian", "clayton", "gumbel", "frank". "
                "Use "independent" for no inter-target correlation."
            )
            lags: Lag order(s) to generate
            calibration_method: "holdout" or "cross"
            calibration_size: Fraction for calibration (from END)
            auto_tune: Whether to tune supported hyperparameters before final fit
            uncertainty_features: Optional hint for heteroscedastic features
            random_state: Random seed
        """
        self.base_model = base_model
        self.horizon = horizon
        self.targets = [targets] if isinstance(targets, str) else targets
        self.copula_family = copula_family
        self.lags = [lags] if isinstance(lags, int) else lags
        self.calibration_method = calibration_method
        self.calibration_size = calibration_size
        self.auto_tune = auto_tune
        self.uncertainty_features = uncertainty_features
        self.random_state = random_state

        # Fitted attributes
        self._fitted = False
        self._copula: BaseCopula | None = None
        self._models_: dict[str, BaseEstimator] = {}
        self._quantiles_: dict[str, np.ndarray] = {}
        self._quantile_levels_: np.ndarray | None = None
        self._feature_cols_: dict[str, list[str]] = {}
        self._uncertainty_drivers_: pl.DataFrame | None = None
        self.tuned_params_: dict[str, float | int] = {}

    def _resolve_quantile_levels(self) -> np.ndarray:
        """Return fit-time quantile levels, with backward-compatible fallback."""
        if self._quantile_levels_ is not None:
            return self._quantile_levels_

        fallback_levels = np.asarray(list(DEFAULT_QUANTILES), dtype=float)
        if self._quantiles_:
            first_target = next(iter(self._quantiles_))
            if len(self._quantiles_[first_target]) != len(fallback_levels):
                raise InvalidDataError(
                    "Current config quantile count does not match fitted residual quantiles. "
                    "Refit the model after setting the desired quantile configuration."
                )
        return fallback_levels

    def _auto_tune(self, data: pl.DataFrame) -> None:
        """Tune params using validation splits, with CV averaging when inner splits exist."""
        eval_splits = build_tune_splits(
            data, task_type="time_series", random_state=self.random_state
        )

        best_score = float("inf")
        best_params: dict[str, float | int] = {}
        best_model = clone(self.base_model)

        for base_params in estimator_param_candidates(self.base_model):
            tuned_base = clone(self.base_model)
            if base_params:
                tuned_base.set_params(**base_params)

            for calib_size in valid_calibration_candidates(
                len(eval_splits[0][0]), self.calibration_size, [0.15, 0.2, 0.25, 0.3]
            ):
                for lags in candidate_values(self.lags[0], [1, 2, 3]):
                    split_scores: list[float] = []
                    for split_train, split_val in eval_splits:
                        candidate = ConformalForecaster(
                            base_model=tuned_base,
                            horizon=self.horizon,
                            targets=self.targets,
                            copula_family=self.copula_family,
                            lags=lags,
                            calibration_method=self.calibration_method,
                            calibration_size=calib_size,
                            auto_tune=False,
                            uncertainty_features=self.uncertainty_features,
                            random_state=self.random_state,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            candidate.fit(split_train)
                            pred = candidate.predict(split_val)
                        actuals = split_val.select(self.targets)
                        score = score_distribution_prediction(
                            pred,
                            actuals,
                            self.targets,
                            confidence=0.9,
                        )
                        split_scores.append(score)
                    avg_score = float(np.mean(split_scores))
                    if avg_score < best_score:
                        best_score = avg_score
                        best_model = clone(tuned_base)
                        best_params = {
                            **base_params,
                            "calibration_size": calib_size,
                            "lags": int(lags),
                        }

        self.base_model = best_model
        self.calibration_size = float(best_params.get("calibration_size", self.calibration_size))
        self.lags = [int(best_params.get("lags", self.lags[0]))]
        self.tuned_params_ = best_params

    def _create_lag_features(
        self,
        data: pl.DataFrame,
        target: str,
    ) -> pl.DataFrame:
        """Create lag features for a target."""
        result = data
        for lag in self.lags:
            result = result.with_columns(pl.col(target).shift(lag).alias(f"{target}_lag_{lag}"))
        return result.drop_nulls()

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> "ConformalForecaster":
        """
        Fit the conformal forecaster.

        Args:
            data: Polars DataFrame or LazyFrame with time series data
            target: Target column name(s) - uses self.targets if not provided
            **kwargs: Additional parameters (unused)

        Returns:
            self (for method chaining)
        """
        # Materialize if needed
        data = materialize_lazyframe(data)

        if self.auto_tune:
            self._auto_tune(data)

        # Create lag features for each target
        data_with_lags = data
        self._quantile_levels_ = np.asarray(list(DEFAULT_QUANTILES), dtype=float)
        for target in self.targets:
            data_with_lags = self._create_lag_features(data_with_lags, target)

        # Automatic validation split strategy selection
        plan = select_validation_plan(
            data_with_lags,
            task_type="time_series",
            random_state=self.random_state,
            holdout_fraction=self.calibration_size,
        )
        train, calib = plan.outer_split

        # Fit per target
        residual_matrix = []

        for target in self.targets:
            feature_cols = [col for col in train.columns if col not in self.targets]
            self._feature_cols_[target] = feature_cols

            x_train = to_numpy(train, feature_cols)
            y_train = to_numpy(train, [target]).flatten()
            x_calib = to_numpy(calib, feature_cols)
            y_calib = to_numpy(calib, [target]).flatten()

            model = clone(self.base_model)
            if self.random_state is not None and "random_state" in model.get_params(deep=False):
                model.set_params(random_state=self.random_state)
            model.fit(x_train, y_train)
            self._models_[target] = model

            calib_preds = model.predict(x_calib)
            residuals = y_calib - calib_preds
            if self._quantile_levels_ is None:
                raise RuntimeError("Internal error: _quantile_levels_ not set before calibration")
            self._quantiles_[target] = np.quantile(residuals, self._quantile_levels_)

            residual_matrix.append(residuals)

        # Fit copula if multivariate
        if len(self.targets) > 1 and self.copula_family != "independent":
            stacked_residuals = np.column_stack(residual_matrix)

            if self.copula_family == "auto":
                selected = auto_select_copula(stacked_residuals)
            elif self.copula_family in COPULA_FAMILIES:
                selected = self.copula_family
            else:
                raise InvalidDataError(
                    f"Unknown copula_family: {self.copula_family}. "
                    f"Valid options: auto, gaussian, clayton, gumbel, frank, independent"
                )

            copula_cls = COPULA_FAMILIES[selected]
            self._copula = copula_cls().fit(stacked_residuals)
        else:
            self._copula = None

        self._fitted = True
        return self

    def predict(
        self,
        data: PolarsInput,
        steps: int | None = None,
    ) -> DistributionPrediction:
        """
        Generate probabilistic forecasts.

        Args:
            data: Polars DataFrame or LazyFrame
            steps: Number of steps to forecast (default: self.horizon)

        Returns:
            DistributionPrediction with quantile forecasts
        """
        if not self._fitted:
            raise ModelNotFittedError("ConformalForecaster")

        steps = steps or self.horizon

        # Materialize if needed
        data = materialize_lazyframe(data)

        # Create lag features
        data_with_lags = data
        for target in self.targets:
            data_with_lags = self._create_lag_features(data_with_lags, target)

        # Get predictions for each target
        quantile_levels = self._resolve_quantile_levels()
        all_quantiles = []
        for target in self.targets:
            x = to_numpy(data_with_lags, self._feature_cols_[target])
            point_preds = self._models_[target].predict(x)

            # Add conformal quantiles
            target_quantiles = self._quantiles_[target]
            if len(target_quantiles) != len(quantile_levels):
                raise InvalidDataError(
                    "Stored quantiles do not match configured quantile levels. "
                    "Refit the model to regenerate compatible quantiles."
                )
            quantile_matrix = np.zeros((len(point_preds), len(quantile_levels)))
            for i, q in enumerate(self._quantiles_[target]):
                quantile_matrix[:, i] = point_preds + q

            all_quantiles.append(quantile_matrix)

        # Stack for multivariate
        if len(self.targets) == 1:
            final_matrix = all_quantiles[0]
        else:
            # Interleave: [target1_q1, target1_q2, ..., target2_q1, target2_q2, ...]
            final_matrix = np.column_stack(
                [
                    all_quantiles[t][:, i]
                    for t in range(len(self.targets))
                    for i in range(len(quantile_levels))
                ]
            )

        return DistributionPrediction(
            quantile_matrix=final_matrix,
            quantile_levels=quantile_levels.tolist(),
            target_names=self.targets,
            copula=self._copula,
        )

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Return residual correlation analysis results."""
        return self._uncertainty_drivers_
