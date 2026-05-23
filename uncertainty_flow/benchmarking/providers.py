"""Provider-based benchmark model seams."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.utils.exceptions import ConfigurationError, ModelNotFittedError
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor

from .configs import ModelBuildConfig
from .model_contracts import BenchmarkModel


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

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if not hasattr(self, "model") or self.model is None:
            raise ModelNotFittedError("BenchmarkModel")
        return self.model.predict(df)


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


@dataclass(frozen=True)
class _DefaultProvider:
    name: str
    adapter_cls: type

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
}


def get_default_providers() -> dict[str, BenchmarkModelProvider]:
    """Return built-in benchmark model providers."""
    return dict(_DEFAULT_PROVIDERS)


class ClassRegistryProvider:
    """Provider that adapts legacy class-registered benchmark models."""

    def __init__(self, name: str, model_cls: type) -> None:
        self.name = name
        self._model_cls = model_cls

    def build(self, config: ModelBuildConfig) -> BenchmarkModel:
        return self._model_cls(_LegacyConfig(config), dict(config.tuned_params or {}))


class _LegacyConfig:
    """Dynamic proxy exposing ModelBuildConfig fields with fallback defaults."""

    _FIELD_DEFAULTS: ClassVar[dict[str, object]] = {
        "n_estimators": 30,
        "horizon": 3,
        "random_state": 42,
        "target_coverage": 0.9,
        "test_size": 0.2,
        "dataset_name": "",
        "n_samples": 1000,
        "auto_tune": True,
        "tune_samples": 500,
        "tune_timeout": 120,
        "confidence_levels": None,
        "dataset_revision": None,
        "hybrid_validation": False,
        "rolling_origin": False,
        "rolling_n_splits": 5,
        "rolling_min_train": 50,
        "rolling_horizon": 1,
        "tune_size": 0.2,
        "target_column": None,
    }

    def __init__(self, config: ModelBuildConfig) -> None:
        self.n_estimators = config.n_estimators
        self.horizon = config.horizon
        self.random_state = config.random_state
        self.target_column = config.target_column
        self.tuned_params = config.tuned_params

    def __getattr__(self, name: str) -> object:
        defaults = _LegacyConfig._FIELD_DEFAULTS
        if name in defaults:
            return defaults[name]
        known = sorted(
            list(defaults)
            + ["n_estimators", "horizon", "random_state", "target_column", "tuned_params"]
        )
        raise AttributeError(f"_LegacyConfig has no attribute '{name}'. Available: {known}")


def resolve_provider(
    model_name: str,
    providers: dict[str, BenchmarkModelProvider],
    class_registry: dict[str, type],
) -> BenchmarkModelProvider:
    """Resolve model provider from explicit providers, then legacy registry."""
    if model_name in providers:
        return providers[model_name]
    if model_name in class_registry:
        return ClassRegistryProvider(model_name, class_registry[model_name])
    raise ConfigurationError(
        f"Unknown model: {model_name}. Available: {sorted(set(providers) | set(class_registry))}"
    )
