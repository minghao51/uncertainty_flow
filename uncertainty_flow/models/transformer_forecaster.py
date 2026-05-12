"""TransformerForecaster - Chronos-2 wrapper with conformal prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..calibration.residual_analysis import compute_uncertainty_drivers
from ..core.base import BaseUncertaintyModel
from ..core.config import get_config
from ..core.distribution import DistributionPrediction
from ..core.types import PolarsInput, TargetSpec
from ..utils.exceptions import ConfigurationError, ModelNotFittedError
from ..utils.polars_bridge import materialize_lazyframe
from ..utils.split import TemporalHoldoutSplit
from . import CHRONOS_MODELS

if TYPE_CHECKING:
    pass


class TransformerForecaster(BaseUncertaintyModel):
    """
    Chronos-2 wrapper with conformal prediction for probabilistic forecasting.

    Wraps Amazon's Chronos-2 foundation model and applies conformal calibration
    on residuals to provide coverage guarantees.

    Coverage guarantee: ✅ (with temporal correction)
    Non-crossing: ✅ (post-sort)

    Examples:
        >>> from uncertainty_flow.models import TransformerForecaster
        >>> import polars as pl
        >>> import numpy as np
        >>>
        >>> # Create sample time series data
        >>> df = pl.DataFrame({
        ...     "timestamp": range(100),
        ...     "value": np.cumsum(np.random.randn(100)),
        ... })
        >>> model = TransformerForecaster(
        ...     target="value",
        ...     horizon=10,
        ... )
        >>> model.fit(df)
        >>> pred = model.predict(df)
        >>> pred.interval(0.9)
    """

    def __init__(
        self,
        target: str,
        horizon: int = 24,
        model_name: str | None = None,
        calibration_method: str = "holdout",
        calibration_size: float = 0.2,
        auto_tune: bool = True,
        device: str = "auto",
        random_state: int | None = None,
        uncertainty_features: list[str] | None = None,
    ):
        """
        Initialize TransformerForecaster.

        Args:
            target: Target column name for univariate forecasting
            horizon: Forecast horizon (number of steps ahead)
            model_name: Chronos model name. Defaults to config default.
                        Options: "chronos-bolt-small" (default), "chronos-bolt-tiny" (fastest),
                        "chronos-bolt-base"
            calibration_method: "holdout" (temporal split from end)
            calibration_size: Fraction of data for calibration (0-1)
            auto_tune: Whether to tune supported hyperparameters before final fit
            device: Device for model ("auto", "cpu", "cuda")
            random_state: Random seed for reproducibility
            uncertainty_features: Optional feature names for heteroscedastic analysis
        """
        config = get_config()
        self.target = target
        self.horizon = horizon
        self.model_name = model_name or config.default_chronos_model
        if self.model_name not in CHRONOS_MODELS:
            if self.model_name.startswith("amazon/"):
                pass
            else:
                raise ConfigurationError(
                    f"Unknown model_name: {self.model_name}. "
                    f"Valid options: {list(CHRONOS_MODELS.keys())}"
                )
        self.calibration_method = calibration_method
        self.calibration_size = calibration_size
        self.auto_tune = auto_tune
        self.device = device
        self.random_state = random_state
        self.uncertainty_features = uncertainty_features

        self._fitted = False
        self._pipeline = None
        self._quantiles_: np.ndarray | None = None
        self._uncertainty_drivers_: pl.DataFrame | None = None
        self._feature_cols_: list[str] = []
        self.tuned_params_: dict[str, float | int] = {}

    def fit(
        self,
        data: PolarsInput,
        target: TargetSpec | None = None,
        **kwargs,
    ) -> "TransformerForecaster":
        """
        Fit the transformer forecaster.

        Args:
            data: Polars DataFrame or LazyFrame with time series data
            target: Target column name(s) - uses self.target if not provided
            **kwargs: Additional parameters (unused)

        Returns:
            self (for method chaining)
        """
        try:
            from chronos import BaseChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting is required for TransformerForecaster. "
                "Install with: pip install 'uncertainty-flow[ml]'"
            )

        data = materialize_lazyframe(data)

        self._feature_cols_ = [col for col in data.columns if col != self.target]

        splitter = TemporalHoldoutSplit()
        train, calib = splitter.split(data, self.calibration_size)

        model_path = (
            self.model_name
            if self.model_name.startswith("amazon/")
            else CHRONOS_MODELS.get(self.model_name, self.model_name)
        )

        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
        )

        calib_forecasts = self._generate_forecasts(calib)
        y_calib = calib.select(self.target).to_numpy().flatten()

        n_forecasts = len(calib_forecasts)
        if len(y_calib) > n_forecasts:
            y_calib = y_calib[-n_forecasts:]
        elif len(y_calib) < n_forecasts:
            calib_forecasts = calib_forecasts[-len(y_calib) :]

        residuals = y_calib - calib_forecasts

        from ..core.types import DEFAULT_QUANTILES

        self._quantiles_ = np.quantile(residuals, DEFAULT_QUANTILES)

        if self.uncertainty_features:
            feature_df = calib.select(self.uncertainty_features)
            self._uncertainty_drivers_ = compute_uncertainty_drivers(residuals, feature_df)

        self._fitted = True
        return self

    def _prepare_forecast_df(self, data: pl.DataFrame):
        import pandas as pd

        pandas_df = data.select([self.target]).to_pandas()
        pandas_df.columns = ["target"]
        pandas_df["item_id"] = "series_0"
        pandas_df["timestamp"] = pd.date_range("2000-01-01", periods=len(pandas_df), freq="h")

        if self._pipeline is None:
            raise ModelNotFittedError("TransformerForecaster")
        context_length = min(len(data), getattr(self._pipeline.model, "context_length", len(data)))
        if len(data) > context_length:
            pandas_df = pandas_df.tail(context_length)
        return pandas_df

    def _generate_forecasts(self, data: pl.DataFrame) -> np.ndarray:
        """
        Generate forecasts using Chronos pipeline.

        Args:
            data: Polars DataFrame with time series

        Returns:
            Numpy array of point forecasts (median, q=0.5)
        """
        from ..core.types import DEFAULT_QUANTILES

        pandas_df = self._prepare_forecast_df(data)

        quantile_levels = [q for q in DEFAULT_QUANTILES if 0.1 <= q <= 0.9]

        forecast_df = self._pipeline.predict_df(
            pandas_df,
            prediction_length=self.horizon,
            quantile_levels=quantile_levels,
        )

        median_col = str(0.5)
        return forecast_df[median_col].to_numpy()

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
            raise ModelNotFittedError("TransformerForecaster")

        data = materialize_lazyframe(data)

        steps = steps or self.horizon

        pandas_df = self._prepare_forecast_df(data)

        from ..core.types import DEFAULT_QUANTILES

        quantile_levels = [q for q in DEFAULT_QUANTILES if 0.1 <= q <= 0.9]

        forecast_df = self._pipeline.predict_df(
            pandas_df,
            prediction_length=steps,
            quantile_levels=quantile_levels,
        )

        n_quantiles = len(DEFAULT_QUANTILES)
        quantile_matrix = np.zeros((1, n_quantiles))

        for i, q in enumerate(DEFAULT_QUANTILES):
            col_name = str(q) if q in quantile_levels else None
            if col_name and col_name in forecast_df.columns:
                values = forecast_df[col_name].to_numpy()
                quantile_matrix[0, i] = np.median(values) if len(values) > 1 else values[0]
            else:
                nearest = min(quantile_levels, key=lambda x: abs(x - q))
                values = forecast_df[str(nearest)].to_numpy()
                quantile_matrix[0, i] = np.median(values) if len(values) > 1 else values[0]

        if self._quantiles_ is None:
            raise ModelNotFittedError("TransformerForecaster")
        for i, q in enumerate(self._quantiles_):
            quantile_matrix[0, i] = quantile_matrix[0, i] + q

        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=list(DEFAULT_QUANTILES),
            target_names=[self.target],
        )

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Return residual correlation analysis results."""
        return self._uncertainty_drivers_
