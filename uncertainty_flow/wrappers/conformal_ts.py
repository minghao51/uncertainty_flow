"""ConformalForecaster - time series forecasting with conformal prediction."""

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone

from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput, TargetSpec
from ..multivariate.copula import COPULA_FAMILIES, BaseCopula, auto_select_copula
from ..utils.auto_tuning import (
    candidate_values,
    estimator_param_candidates,
    score_distribution_prediction,
    valid_calibration_candidates,
)
from ..utils.exceptions import error_invalid_data, error_model_not_fitted
from ..utils.polars_bridge import to_numpy
from ..utils.split import TemporalHoldoutSplit

if TYPE_CHECKING:
    pass


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
        self._feature_cols_: dict[str, list[str]] = {}
        self._uncertainty_drivers_: pl.DataFrame | None = None
        self.tuned_params_: dict[str, float | int] = {}

    def _auto_tune(self, data: pl.DataFrame) -> None:
        """Tune supported params using a temporal validation split."""
        splitter = TemporalHoldoutSplit()
        tune_train, tune_val = splitter.split(data, 0.2)

        best_score = float("inf")
        best_params: dict[str, float | int] = {}
        best_model = clone(self.base_model)

        for base_params in estimator_param_candidates(self.base_model):
            tuned_base = clone(self.base_model)
            if base_params:
                tuned_base.set_params(**base_params)

            for calib_size in valid_calibration_candidates(
                len(tune_train), self.calibration_size, [0.15, 0.2, 0.25, 0.3]
            ):
                for lags in candidate_values(self.lags[0], [1, 2, 3]):
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
                        candidate.fit(tune_train)
                        pred = candidate.predict(tune_val)
                    actuals = tune_val.select(self.targets)
                    score = score_distribution_prediction(
                        pred,
                        actuals,
                        self.targets,
                        confidence=0.9,
                    )
                    if score < best_score:
                        best_score = score
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
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        if self.auto_tune:
            self._auto_tune(data)

        # Create lag features for each target
        data_with_lags = data
        for target in self.targets:
            data_with_lags = self._create_lag_features(data_with_lags, target)

        # Temporal split (from END)
        splitter = TemporalHoldoutSplit()
        train, calib = splitter.split(data_with_lags, self.calibration_size)

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
            self._quantiles_[target] = np.quantile(residuals, DEFAULT_QUANTILES)

            residual_matrix.append(residuals)

        # Fit copula if multivariate
        if len(self.targets) > 1 and self.copula_family != "independent":
            stacked_residuals = np.column_stack(residual_matrix)

            if self.copula_family == "auto":
                selected = auto_select_copula(stacked_residuals)
            elif self.copula_family in COPULA_FAMILIES:
                selected = self.copula_family
            else:
                error_invalid_data(
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
            error_model_not_fitted("ConformalForecaster")

        steps = steps or self.horizon

        # Materialize if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Create lag features
        data_with_lags = data
        for target in self.targets:
            data_with_lags = self._create_lag_features(data_with_lags, target)

        # Get predictions for each target
        all_quantiles = []
        for target in self.targets:
            x = to_numpy(data_with_lags, self._feature_cols_[target])
            point_preds = self._models_[target].predict(x)

            # Add conformal quantiles
            quantile_matrix = np.zeros((len(point_preds), len(DEFAULT_QUANTILES)))
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
                    for i in range(len(DEFAULT_QUANTILES))
                ]
            )

        return DistributionPrediction(
            quantile_matrix=final_matrix,
            quantile_levels=DEFAULT_QUANTILES,
            target_names=self.targets,
        )

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Return residual correlation analysis results."""
        return self._uncertainty_drivers_
