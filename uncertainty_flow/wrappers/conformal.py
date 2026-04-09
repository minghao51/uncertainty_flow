"""ConformalRegressor - wrap any sklearn model with conformal prediction."""

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone

from ..calibration.residual_analysis import compute_uncertainty_drivers
from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput, TargetSpec
from ..utils.auto_tuning import (
    estimator_param_candidates,
    score_distribution_prediction,
    valid_calibration_candidates,
)
from ..utils.exceptions import ConfigurationError, error_model_not_fitted
from ..utils.polars_bridge import (
    materialize_lazyframe,
    to_numpy_series_zero_copy,
    to_numpy_zero_copy,
)
from ..utils.split import RandomHoldoutSplit

if TYPE_CHECKING:
    pass


class ConformalRegressor(BaseUncertaintyModel):
    """
    Wrap any scikit-learn regressor with conformal prediction.

    Coverage guarantee: ✅ (exchangeability assumption)
    Non-crossing: ✅ (post-sort)

    Examples:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from uncertainty_flow.wrappers import ConformalRegressor
        >>> import polars as pl
        >>>
        >>> df = pl.DataFrame({
        ...     "feature1": [1, 2, 3, 4, 5],
        ...     "feature2": [2, 4, 6, 8, 10],
        ...     "target": [1.5, 3.5, 5.5, 7.5, 9.5],
        ... })
        >>> base = GradientBoostingRegressor(random_state=42)
        >>> model = ConformalRegressor(base_model=base, random_state=42)
        >>> model.fit(df, target="target")
        >>> pred = model.predict(df)
        >>> pred.interval(0.9)
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        calibration_method: str = "holdout",
        calibration_size: float = 0.2,
        coverage_target: float = 0.9,
        auto_tune: bool = True,
        uncertainty_features: list[str] | None = None,
        random_state: int | None = None,
    ):
        """
        Initialize ConformalRegressor.

        Args:
            base_model: Any sklearn-compatible regressor
            calibration_method: "holdout" or "cross"
            calibration_size: Fraction of data for calibration (0-1)
            coverage_target: Default coverage level for intervals
            auto_tune: Whether to tune supported hyperparameters before final fit
            uncertainty_features: Optional hint for heteroscedastic features
            random_state: Random seed
        """
        self.base_model = base_model
        self.calibration_method = calibration_method
        self.calibration_size = calibration_size
        self.coverage_target = coverage_target
        self.auto_tune = auto_tune
        self.uncertainty_features = uncertainty_features
        self.random_state = random_state

        # Fitted attributes
        self._fitted = False
        self._feature_cols_: list[str] = []
        self._target_col_: str = ""
        self._quantiles_: np.ndarray | None = None
        self._uncertainty_drivers_: pl.DataFrame | None = None
        self.tuned_params_: dict[str, float | int] = {}

    def _auto_tune(
        self,
        data: pl.DataFrame,
        target: str,
    ) -> None:
        """Tune supported params using a validation split."""
        splitter = RandomHoldoutSplit(random_state=self.random_state)
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
                candidate = ConformalRegressor(
                    base_model=tuned_base,
                    calibration_method=self.calibration_method,
                    calibration_size=calib_size,
                    coverage_target=self.coverage_target,
                    auto_tune=False,
                    uncertainty_features=self.uncertainty_features,
                    random_state=self.random_state,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidate.fit(tune_train, target=target)
                    pred = candidate.predict(tune_val)
                score = score_distribution_prediction(
                    pred,
                    tune_val[target],
                    [target],
                    confidence=self.coverage_target,
                )
                if score < best_score:
                    best_score = score
                    best_model = clone(tuned_base)
                    best_params = {**base_params, "calibration_size": calib_size}

        self.base_model = best_model
        self.calibration_size = float(best_params.get("calibration_size", self.calibration_size))
        self.tuned_params_ = best_params

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> "ConformalRegressor":
        """
        Fit the conformal regressor.

        Args:
            data: Polars DataFrame or LazyFrame with features and target
            target: Target column name(s)
            **kwargs: Additional parameters (unused)

        Returns:
            self (for method chaining)
        """
        # Materialize LazyFrame once at the start
        data = materialize_lazyframe(data)

        # Get feature columns
        if target is None:
            raise ConfigurationError("target is required for ConformalRegressor")
        target_str = target if isinstance(target, str) else target[0]
        self._target_col_ = target_str

        if self.auto_tune:
            self._auto_tune(data, target_str)

        feature_cols = [col for col in data.columns if col != target_str]
        self._feature_cols_ = feature_cols

        # Split data
        splitter = RandomHoldoutSplit(random_state=self.random_state)
        train, calib = splitter.split(data, self.calibration_size)

        # Convert to numpy - single collect already done above
        x_train = to_numpy_zero_copy(train, feature_cols)
        y_train = to_numpy_series_zero_copy(train[target_str]).flatten()
        x_calib = to_numpy_zero_copy(calib, feature_cols)
        y_calib = to_numpy_series_zero_copy(calib[target_str]).flatten()

        # Fit base model
        self.base_model.fit(x_train, y_train)

        # Compute conformal quantiles on calibration set
        calib_preds = self.base_model.predict(x_calib)
        residuals = y_calib - calib_preds

        # Store quantiles for prediction
        self._quantiles_ = np.quantile(residuals, DEFAULT_QUANTILES)

        # Compute uncertainty drivers using calibration features
        calib_features = calib.select(feature_cols)
        self._uncertainty_drivers_ = compute_uncertainty_drivers(residuals, calib_features)

        self._fitted = True
        return self

    def predict(self, data: PolarsInput) -> DistributionPrediction:
        """
        Generate probabilistic predictions.

        Args:
            data: Polars DataFrame or LazyFrame with features

        Returns:
            DistributionPrediction with quantile predictions
        """
        if not self._fitted:
            error_model_not_fitted("ConformalRegressor")

        # Materialize LazyFrame if needed
        data = materialize_lazyframe(data)

        # Get predictions
        x = to_numpy_zero_copy(data, self._feature_cols_)
        point_preds = self.base_model.predict(x)

        # Add conformal quantiles
        assert self._quantiles_ is not None
        quantile_matrix = np.zeros((len(point_preds), len(DEFAULT_QUANTILES)))
        for i, q in enumerate(self._quantiles_):
            quantile_matrix[:, i] = point_preds + q

        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=DEFAULT_QUANTILES,
            target_names=[self._target_col_],
        )

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Return residual correlation analysis results."""
        return self._uncertainty_drivers_
