"""ConformalRegressor - wrap any sklearn model with conformal prediction."""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from ..calibration.residual_analysis import compute_uncertainty_drivers
from ..core.base import BaseUncertaintyModel
from ..core.distribution import DistributionPrediction
from ..core.types import DEFAULT_QUANTILES, PolarsInput, TargetSpec
from ..utils.exceptions import error_model_not_fitted
from ..utils.polars_bridge import to_numpy
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
            uncertainty_features: Optional hint for heteroscedastic features
            random_state: Random seed
        """
        self.base_model = base_model
        self.calibration_method = calibration_method
        self.calibration_size = calibration_size
        self.coverage_target = coverage_target
        self.uncertainty_features = uncertainty_features
        self.random_state = random_state

        # Fitted attributes
        self._fitted = False
        self._feature_cols_: list[str] = []
        self._target_col_: str = ""
        self._quantiles_: np.ndarray | None = None
        self._uncertainty_drivers_: pl.DataFrame | None = None

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
        # Materialize LazyFrame if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Get feature columns
        if target is None:
            raise ValueError("target is required for ConformalRegressor")
        target_str = target if isinstance(target, str) else target[0]
        self._target_col_ = target_str
        self._feature_cols_ = [col for col in data.columns if col != target_str]

        # Split data
        splitter = RandomHoldoutSplit(random_state=self.random_state)
        train, calib = splitter.split(data, self.calibration_size)

        # Convert to numpy
        x_train = to_numpy(train, self._feature_cols_)
        y_train = to_numpy(train, [target_str]).flatten()
        x_calib = to_numpy(calib, self._feature_cols_)
        y_calib = to_numpy(calib, [target_str]).flatten()

        # Fit base model
        self.base_model.fit(x_train, y_train)

        # Compute conformal quantiles on calibration set
        calib_preds = self.base_model.predict(x_calib)
        residuals = y_calib - calib_preds

        # Store quantiles for prediction
        self._quantiles_ = np.quantile(residuals, DEFAULT_QUANTILES)

        # Compute uncertainty drivers using calibration features
        feature_df = calib.select(self._feature_cols_)
        self._uncertainty_drivers_ = compute_uncertainty_drivers(residuals, feature_df)

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
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Get predictions
        x = to_numpy(data, self._feature_cols_)
        point_preds = self.base_model.predict(x)

        # Add conformal quantiles
        assert self._quantiles_ is not None
        quantile_matrix = np.zeros((len(point_preds), len(DEFAULT_QUANTILES)))
        for i, q in enumerate(self._quantiles_):
            quantile_matrix[:, i] = point_preds + q

        # Extract index if available
        index = None
        if data.height == len(point_preds):
            try:
                index = data.select(pl.row_index()).to_series()
            except Exception:
                pass

        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=DEFAULT_QUANTILES,
            target_names=[self._target_col_],
            index=index,
        )

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Return residual correlation analysis results."""
        return self._uncertainty_drivers_
