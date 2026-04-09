"""Benchmark runner for uncertainty_flow models with auto-tuning."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.metrics import coverage_score, pinball_loss, winkler_score
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.utils.exceptions import (
    ConfigurationError,
    DataError,
    ModelNotFittedError,
)
from uncertainty_flow.utils.polars_bridge import to_numpy_series_zero_copy
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor

from .datasets import DatasetInfo, load_dataset
from .tuning import TuningConfig, auto_tune_model

if TYPE_CHECKING:
    pass

SEARCH_SPACE = {
    "quantile-forest": {
        "n_estimators": [20, 30, 50],
        "horizon": [2, 3, 5],
    },
    "conformal-regressor": {
        "n_estimators": [20, 30, 50],
        "calibration_size": [0.15, 0.20, 0.25, 0.30],
    },
    "conformal-forecaster": {
        "n_estimators": [20, 30, 50],
        "calibration_size": [0.15, 0.20, 0.25],
        "lags": [1, 2, 3],
    },
}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        dataset_name: Name of the dataset to benchmark
        n_samples: Number of samples to use (default: 1000)
        horizon: Forecast horizon for time series models (default: 3)
        n_estimators: Number of base estimators (default: 30)
        confidence_levels: Coverage levels to evaluate (default: [0.8, 0.9, 0.95])
        random_state: Random seed for reproducibility (default: 42)
        target_column: Target column name (auto-detected if None)
        auto_tune: Whether to auto-tune hyperparameters (default: True)
        target_coverage: Target coverage level for tuning (default: 0.9)
        tune_samples: Number of samples to use for tuning (default: 500)
        tune_timeout: Max seconds per model for tuning (default: 120)
    """

    dataset_name: str
    n_samples: int = 1000
    horizon: int = 3
    n_estimators: int = 30
    confidence_levels: list[float] | None = None
    random_state: int = 42
    target_column: str | None = None
    auto_tune: bool = True
    target_coverage: float = 0.9
    tune_samples: int = 500
    tune_timeout: int = 120

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.8, 0.9, 0.95]


@dataclass
class ModelResult:
    """Results for a single model on a dataset."""

    model_name: str
    coverage_90: float
    coverage_80: float
    sharpness_90: float
    sharpness_80: float
    winkler_90: float
    winkler_80: float
    pinball_loss: float
    train_time_sec: float
    n_samples: int
    tuned_params: dict[str, Any] = field(default_factory=dict)
    was_tuned: bool = False


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a dataset."""

    run_id: str
    timestamp: str
    dataset_name: str
    dataset_domain: str
    n_samples: int
    horizon: int
    models: list[ModelResult]


MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    """Decorator to register a model for benchmarking."""

    def decorator(cls: type) -> type:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


@register_model("quantile-forest")
class QuantileForestBenchmark:
    """Benchmark wrapper for QuantileForestForecaster."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict[str, Any] | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.model: QuantileForestForecaster | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        n_est = self.tuned_params.get("n_estimators", self.config.n_estimators)
        horizon = self.tuned_params.get("horizon", self.config.horizon)

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

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("BenchmarkModel")
        return self.model.predict(df)


@register_model("conformal-regressor")
class ConformalRegressorBenchmark:
    """Benchmark wrapper for ConformalRegressor."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict[str, Any] | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.model: ConformalRegressor | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        n_est = self.tuned_params.get("n_estimators", self.config.n_estimators)
        calib = self.tuned_params.get("calibration_size", 0.2)

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

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("BenchmarkModel")
        return self.model.predict(df)


@register_model("conformal-forecaster")
class ConformalForecasterBenchmark:
    """Benchmark wrapper for ConformalForecaster."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict[str, Any] | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.model: ConformalForecaster | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        n_est = self.tuned_params.get("n_estimators", self.config.n_estimators)
        calib = self.tuned_params.get("calibration_size", 0.2)
        lags = self.tuned_params.get("lags", 2)

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

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise ModelNotFittedError("BenchmarkModel")
        return self.model.predict(df)


class BenchmarkRunner:
    """Runner for executing benchmarks on datasets with optional auto-tuning.

    Example:
        >>> config = BenchmarkConfig(dataset_name="weather", n_samples=1000)
        >>> runner = BenchmarkRunner(config)
        >>> runner.load_data()
        >>> result = runner.run_all()
        >>> print(result.models[0].coverage_90)
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.df: pl.DataFrame | None = None
        self.ds_info: DatasetInfo | None = None
        self.results: list[ModelResult] = []
        self._run_result: BenchmarkResult | None = None
        self._tuning_cache: dict[str, dict[str, Any]] = {}

    def load_data(self) -> None:
        """Load the dataset."""
        self.df, self.ds_info = load_dataset(
            self.config.dataset_name,
            n_samples=self.config.n_samples,
        )
        if self.config.target_column:
            self.target = self.config.target_column
        else:
            self.target = self.ds_info.default_target

    def _get_tuned_params(self, model_name: str) -> dict[str, Any]:
        """Get tuned parameters for a model, running tuning if needed."""
        if model_name in self._tuning_cache:
            return self._tuning_cache[model_name]

        if not self.config.auto_tune:
            return {}

        if self.df is None:
            raise DataError("Data not loaded. Call load_data() first.")

        print(f"    Auto-tuning {model_name}...")
        tune_config = TuningConfig(
            target_coverage=self.config.target_coverage,
            n_samples=self.config.tune_samples,
            timeout=self.config.tune_timeout,
        )

        tuning_result = auto_tune_model(
            model_name=model_name,
            df=self.df,
            target=self.target,
            horizon=self.config.horizon,
            config=tune_config,
        )

        self._tuning_cache[model_name] = tuning_result.best_params
        return tuning_result.best_params

    def run_model(self, model_name: str) -> ModelResult:
        """Run a single model benchmark with optional auto-tuning."""
        if model_name not in MODEL_REGISTRY:
            raise ConfigurationError(
                f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        if self.df is None:
            raise DataError("Data not loaded. Call load_data() first.")

        tuned_params = self._get_tuned_params(model_name)
        was_tuned = bool(tuned_params)

        if was_tuned:
            print(f"    Using tuned params: {tuned_params}")

        model_cls = MODEL_REGISTRY[model_name]
        benchmark = model_cls(self.config, tuned_params)

        print(f"    Fitting {model_name}...")
        benchmark.fit(self.df, self.target)

        print("    Predicting...")
        pred = benchmark.predict(self.df)

        interval_90 = pred.interval(0.9)
        interval_80 = pred.interval(0.8)

        n_pred = len(interval_90)
        y_true = to_numpy_series_zero_copy(self.df[self.target])[-n_pred:]
        lower_90 = to_numpy_series_zero_copy(
            interval_90[
                "lower" if len(pred._targets) == 1 else f"{pred._targets[0]}_lower"
            ]
        )
        upper_90 = to_numpy_series_zero_copy(
            interval_90[
                "upper" if len(pred._targets) == 1 else f"{pred._targets[0]}_upper"
            ]
        )
        lower_80 = to_numpy_series_zero_copy(
            interval_80[
                "lower" if len(pred._targets) == 1 else f"{pred._targets[0]}_lower"
            ]
        )
        upper_80 = to_numpy_series_zero_copy(
            interval_80[
                "upper" if len(pred._targets) == 1 else f"{pred._targets[0]}_upper"
            ]
        )

        cov_90 = coverage_score(y_true, lower_90, upper_90)
        cov_80 = coverage_score(y_true, lower_80, upper_80)
        wink_90 = winkler_score(y_true, lower_90, upper_90, confidence=0.9)
        wink_80 = winkler_score(y_true, lower_80, upper_80, confidence=0.8)
        sharp_90 = float(np.mean(upper_90 - lower_90))
        sharp_80 = float(np.mean(upper_80 - lower_80))

        pinball = pinball_loss(
            y_true,
            lower_90,
            0.1,
        )

        return ModelResult(
            model_name=model_name,
            coverage_90=round(cov_90, 4),
            coverage_80=round(cov_80, 4),
            sharpness_90=round(sharp_90, 4),
            sharpness_80=round(sharp_80, 4),
            winkler_90=round(wink_90, 4),
            winkler_80=round(wink_80, 4),
            pinball_loss=round(float(pinball), 4),
            train_time_sec=round(benchmark.train_time, 3),
            n_samples=n_pred,
            tuned_params=tuned_params,
            was_tuned=was_tuned,
        )

    def run_all(self, model_names: list[str] | None = None) -> BenchmarkResult:
        """Run all benchmarks for configured dataset.

        Args:
            model_names: List of model names to run. If None, runs all registered.

        Returns:
            BenchmarkResult with all model results
        """
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        self.results = []

        for model_name in model_names:
            try:
                result = self.run_model(model_name)
                self.results.append(result)
            except Exception as e:
                print(f"    ERROR running {model_name}: {e}")

        self._run_result = BenchmarkResult(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=self.config.dataset_name,
            dataset_domain=self.ds_info.domain if self.ds_info else "Unknown",
            n_samples=self.config.n_samples,
            horizon=self.config.horizon,
            models=self.results,
        )

        return self._run_result

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        if self._run_result is None:
            return {"metadata": {}, "results": []}
        return {
            "metadata": {
                "run_id": self._run_result.run_id,
                "timestamp": self._run_result.timestamp,
                "dataset": self._run_result.dataset_name,
                "domain": self._run_result.dataset_domain,
                "n_samples": self._run_result.n_samples,
                "horizon": self._run_result.horizon,
                "auto_tune": self.config.auto_tune,
                "target_coverage": self.config.target_coverage,
            },
            "results": [
                {
                    "model": r.model_name,
                    "coverage_90": r.coverage_90,
                    "coverage_80": r.coverage_80,
                    "sharpness_90": r.sharpness_90,
                    "sharpness_80": r.sharpness_80,
                    "winkler_90": r.winkler_90,
                    "winkler_80": r.winkler_80,
                    "pinball_loss": r.pinball_loss,
                    "train_time_sec": r.train_time_sec,
                    "n_samples": r.n_samples,
                    "tuned_params": r.tuned_params,
                    "was_tuned": r.was_tuned,
                }
                for r in self._run_result.models
            ],
        }

    def save_json(self, path: Path | str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: Path | str) -> None:
        """Save results to CSV file."""
        if not self.results:
            return

        rows = []
        for r in self.results:
            rows.append(
                {
                    "dataset": self.config.dataset_name,
                    "domain": self.ds_info.domain if self.ds_info else "Unknown",
                    "model": r.model_name,
                    "n_samples": r.n_samples,
                    "horizon": self.config.horizon,
                    "coverage_90": r.coverage_90,
                    "coverage_80": r.coverage_80,
                    "sharpness_90": r.sharpness_90,
                    "sharpness_80": r.sharpness_80,
                    "winkler_90": r.winkler_90,
                    "winkler_80": r.winkler_80,
                    "pinball_loss": r.pinball_loss,
                    "train_time_sec": r.train_time_sec,
                    "was_tuned": r.was_tuned,
                    "tuned_params": str(r.tuned_params),
                }
            )

        df = pl.DataFrame(rows)
        df.write_csv(path)
