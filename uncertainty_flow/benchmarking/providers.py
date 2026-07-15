"""Provider-based benchmark model seams."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, cast

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.utils.exceptions import ConfigurationError, ModelNotFittedError
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor

from .model_contracts import BenchmarkModel, ModelBuildConfig


def _int_param(params: dict[str, object] | None, key: str, default: int) -> int:
    if params is None:
        return default
    value = params.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return int(value)
    return default


def _float_param(params: dict[str, object] | None, key: str, default: float) -> float:
    if params is None:
        return default
    value = params.get(key, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | str):
        return float(value)
    return default


def _hidden_layers(params: dict[str, object]) -> tuple[int, ...]:
    value = params.get("hidden_layer_sizes", (64, 32))
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return tuple(value)
    return (64, 32)


class BenchmarkModelProvider(Protocol):
    """Provider interface for concrete benchmark model adapters."""

    name: str

    def build(self, config: ModelBuildConfig) -> BenchmarkModel:
        """Build a model adapter instance."""


@dataclass
class _BaseAdapter:
    """Shared base for benchmark model adapters."""

    config: ModelBuildConfig
    train_time: float = 0.0
    target: str | None = None

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if not hasattr(self, "model") or self.model is None:
            raise ModelNotFittedError("BenchmarkModel")
        return self.model.predict(df)


def _prediction(
    lower: np.ndarray, median: np.ndarray, upper: np.ndarray, target: str
) -> DistributionPrediction:
    """Build the common quantile representation for deterministic baselines."""

    return DistributionPrediction(
        np.column_stack((lower, median, upper)),
        [0.1, 0.5, 0.9],
        [target],
    )


@dataclass
class _QuantileForestAdapter(_BaseAdapter):
    model: QuantileForestForecaster | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        n_est = _int_param(self.config.tuned_params, "n_estimators", self.config.n_estimators)
        horizon = _int_param(self.config.tuned_params, "horizon", self.config.horizon)
        self.model = QuantileForestForecaster(
            targets=target,
            horizon=horizon,
            n_estimators=n_est,
            auto_tune=False,
            random_state=self.config.random_state,
        )
        start = time.time()
        self.model.fit(df)
        self.train_time = time.time() - start


@dataclass
class _ConformalRegressorAdapter(_BaseAdapter):
    model: ConformalRegressor | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        n_est = _int_param(self.config.tuned_params, "n_estimators", self.config.n_estimators)
        calib = _float_param(self.config.tuned_params, "calibration_size", 0.2)
        self.model = ConformalRegressor(
            base_model=GradientBoostingRegressor(
                n_estimators=n_est,
                random_state=self.config.random_state,
            ),
            calibration_size=calib,
            auto_tune=False,
            random_state=self.config.random_state,
        )
        start = time.time()
        self.model.fit(df, target=target)
        self.train_time = time.time() - start


@dataclass
class _ConformalForecasterAdapter(_BaseAdapter):
    model: ConformalForecaster | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        n_est = _int_param(self.config.tuned_params, "n_estimators", self.config.n_estimators)
        calib = _float_param(self.config.tuned_params, "calibration_size", 0.2)
        lags = _int_param(self.config.tuned_params, "lags", 2)
        self.model = ConformalForecaster(
            base_model=GradientBoostingRegressor(
                n_estimators=n_est,
                random_state=self.config.random_state,
            ),
            horizon=self.config.horizon,
            targets=target,
            lags=lags,
            calibration_size=calib,
            auto_tune=False,
            random_state=self.config.random_state,
        )
        start = time.time()
        self.model.fit(df)
        self.train_time = time.time() - start


@dataclass
class _ConformalBaselineAdapter(_BaseAdapter):
    model: ConformalRegressor | None = None
    estimator_kind: str = "linear"

    def fit(self, df: pl.DataFrame, target: str) -> None:
        params = self.config.tuned_params or {}
        calibration_size = _float_param(params, "calibration_size", 0.2)
        n_estimators = _int_param(params, "n_estimators", self.config.n_estimators)
        if self.estimator_kind == "linear":
            estimator = LinearRegression()
        elif self.estimator_kind == "ridge":
            alpha = _float_param(params, "alpha", 1.0)
            estimator = Ridge(alpha=alpha)
        elif self.estimator_kind == "random-forest":
            estimator = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            estimator = GradientBoostingRegressor(
                n_estimators=n_estimators,
                random_state=self.config.random_state,
            )
        self.model = ConformalRegressor(
            base_model=estimator,
            calibration_size=calibration_size,
            auto_tune=False,
            random_state=self.config.random_state,
        )
        start = time.perf_counter()
        self.model.fit(df, target=target)
        self.target = target
        self.train_time = time.perf_counter() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("ConformalBaseline")
        return self.model.predict(df)


@dataclass
class _NaiveForecastAdapter(_BaseAdapter):
    last_value: float | None = None
    residual_std: float | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        start = time.perf_counter()
        values = df[target].to_numpy()
        self.last_value = float(values[-1])
        horizon = _int_param(self.config.tuned_params, "horizon", self.config.horizon)
        self.residual_std = (
            float(np.std(np.diff(values)) * np.sqrt(horizon))
            if len(values) > 1
            else float(np.std(values))
        )
        self.target = target
        self.train_time = time.perf_counter() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.last_value is None or self.residual_std is None or self.target is None:
            raise ModelNotFittedError("NaiveForecast")
        n = len(df)
        return _prediction(
            np.full(n, self.last_value - 1.645 * self.residual_std),
            np.full(n, self.last_value),
            np.full(n, self.last_value + 1.645 * self.residual_std),
            self.target,
        )


@dataclass
class _MovingAverageAdapter(_BaseAdapter):
    average: float | None = None
    residual_std: float | None = None
    window: int = 5

    def fit(self, df: pl.DataFrame, target: str) -> None:
        start = time.perf_counter()
        values = df[target].to_numpy()
        self.window = _int_param(self.config.tuned_params, "window", 5)
        self.average = float(np.mean(values[-self.window :]))
        if len(values) > self.window:
            residuals = values[self.window :] - np.array(
                [np.mean(values[i - self.window : i]) for i in range(self.window, len(values))]
            )
            self.residual_std = float(np.std(residuals))
        else:
            self.residual_std = float(np.std(values))
        self.target = target
        self.train_time = time.perf_counter() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.average is None or self.residual_std is None or self.target is None:
            raise ModelNotFittedError("MovingAverage")
        n = len(df)
        return _prediction(
            np.full(n, self.average - 1.645 * self.residual_std),
            np.full(n, self.average),
            np.full(n, self.average + 1.645 * self.residual_std),
            self.target,
        )


@dataclass
class _DeepQuantileAdapter(_BaseAdapter):
    model: object | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        from uncertainty_flow.models import DeepQuantileNet

        params = self.config.tuned_params or {}
        self.model = DeepQuantileNet(
            hidden_layer_sizes=_hidden_layers(params),
            trunk_max_iter=_int_param(params, "trunk_max_iter", 300),
            random_state=self.config.random_state,
        )
        start = time.perf_counter()
        self.model.fit(df, target=target)
        self.target = target
        self.train_time = time.perf_counter() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("DeepQuantileNet")
        return cast(DistributionPrediction, self.model.predict(df))


@dataclass
class _DeepQuantileTorchAdapter(_BaseAdapter):
    model: object | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        try:
            from uncertainty_flow.models import DeepQuantileNetTorch
        except ImportError as error:
            raise ImportError(
                "torch required for deep-quantile-torch. Install: uv sync --extra opinion"
            ) from error
        params = self.config.tuned_params or {}
        self.model = DeepQuantileNetTorch(
            hidden_layer_sizes=_hidden_layers(params),
            epochs=_int_param(params, "epochs", 100),
            learning_rate=_float_param(params, "learning_rate", 0.001),
            device="cpu",
            random_state=self.config.random_state,
        )
        start = time.perf_counter()
        self.model.fit(df, target=target)
        self.target = target
        self.train_time = time.perf_counter() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("DeepQuantileNetTorch")
        return cast(DistributionPrediction, self.model.predict(df))


@dataclass
class _TransformerForecasterAdapter(_BaseAdapter):
    model: object | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        try:
            from uncertainty_flow.models import TransformerForecaster
        except ImportError as error:
            raise ImportError(
                "chronos-forecasting required. Install: uv sync --extra opinion"
            ) from error
        params = self.config.tuned_params or {}
        self.model = TransformerForecaster(
            target=target,
            horizon=self.config.horizon,
            model_name=str(params.get("chronos_model", "chronos-bolt-tiny")),
            calibration_size=_float_param(params, "calibration_size", 0.2),
            auto_tune=False,
            device="cpu",
            random_state=self.config.random_state,
        )
        start = time.perf_counter()
        self.model.fit(df)
        self.target = target
        self.train_time = time.perf_counter() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("TransformerForecaster")
        return cast(DistributionPrediction, self.model.predict(df))


@dataclass
class _BayesianQuantileAdapter(_BaseAdapter):
    model: object | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        try:
            from uncertainty_flow.bayesian import BayesianQuantileRegressor
        except ImportError as error:
            raise ImportError(
                "numpyro is required. Install with: uv sync --extra opinion"
            ) from error
        params = self.config.tuned_params or {}
        self.model = BayesianQuantileRegressor(
            n_warmup=_int_param(params, "n_warmup", 1000),
            n_samples=_int_param(params, "n_samples", 2000),
            prior_width=_float_param(params, "prior_width", 10.0),
            random_state=self.config.random_state,
        )
        start = time.perf_counter()
        self.model.fit(df, target=target)
        self.target = target
        self.train_time = time.perf_counter() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("BayesianQuantileRegressor")
        return cast(DistributionPrediction, self.model.predict(df))


@dataclass(frozen=True)
class _DefaultProvider:
    name: str
    adapter_cls: Callable[[ModelBuildConfig], BenchmarkModel]

    def build(self, config: ModelBuildConfig) -> BenchmarkModel:
        return self.adapter_cls(config)


_DEFAULT_PROVIDERS: dict[str, BenchmarkModelProvider] = {
    "quantile-forest": cast(
        BenchmarkModelProvider, _DefaultProvider("quantile-forest", _QuantileForestAdapter)
    ),
    "conformal-regressor": cast(
        BenchmarkModelProvider,
        _DefaultProvider("conformal-regressor", _ConformalRegressorAdapter),
    ),
    "conformal-forecaster": cast(
        BenchmarkModelProvider,
        _DefaultProvider("conformal-forecaster", _ConformalForecasterAdapter),
    ),
    "linear-regression": cast(
        BenchmarkModelProvider,
        _DefaultProvider(
            "linear-regression",
            lambda config: _ConformalBaselineAdapter(config, estimator_kind="linear"),
        ),
    ),
    "ridge-regression": cast(
        BenchmarkModelProvider,
        _DefaultProvider(
            "ridge-regression",
            lambda config: _ConformalBaselineAdapter(config, estimator_kind="ridge"),
        ),
    ),
    "random-forest": cast(
        BenchmarkModelProvider,
        _DefaultProvider(
            "random-forest",
            lambda config: _ConformalBaselineAdapter(config, estimator_kind="random-forest"),
        ),
    ),
    "gradient-boosting": cast(
        BenchmarkModelProvider,
        _DefaultProvider(
            "gradient-boosting",
            lambda config: _ConformalBaselineAdapter(config, estimator_kind="gradient-boosting"),
        ),
    ),
    "naive-forecast": cast(
        BenchmarkModelProvider, _DefaultProvider("naive-forecast", _NaiveForecastAdapter)
    ),
    "moving-average": cast(
        BenchmarkModelProvider, _DefaultProvider("moving-average", _MovingAverageAdapter)
    ),
    "deep-quantile": cast(
        BenchmarkModelProvider, _DefaultProvider("deep-quantile", _DeepQuantileAdapter)
    ),
    "deep-quantile-torch": cast(
        BenchmarkModelProvider,
        _DefaultProvider("deep-quantile-torch", _DeepQuantileTorchAdapter),
    ),
    "transformer-forecaster": cast(
        BenchmarkModelProvider,
        _DefaultProvider("transformer-forecaster", _TransformerForecasterAdapter),
    ),
    "bayesian-quantile": cast(
        BenchmarkModelProvider,
        _DefaultProvider("bayesian-quantile", _BayesianQuantileAdapter),
    ),
}


def get_default_providers() -> dict[str, BenchmarkModelProvider]:
    """Return built-in benchmark model providers."""
    return dict(_DEFAULT_PROVIDERS)


def resolve_provider(
    model_name: str,
    providers: dict[str, BenchmarkModelProvider],
    class_registry: dict[str, type],
) -> BenchmarkModelProvider:
    """Resolve a supported provider while rejecting retired class registrations."""

    del class_registry
    try:
        return providers[model_name]
    except KeyError as error:
        raise ConfigurationError(
            f"Unknown model: {model_name}. Available: {sorted(providers)}"
        ) from error
