"""Benchmark runner for uncertainty_flow models with auto-tuning."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from uncertainty_flow.utils.polars_bridge import to_numpy_series
from uncertainty_flow.utils.split import RollingOriginSplit
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor

from .datasets import DatasetInfo, load_dataset
from .tuning import TuningConfig, auto_tune_model

logger = logging.getLogger(__name__)


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
    test_size: float = 0.2
    dataset_revision: str | None = None
    hybrid_validation: bool = False
    rolling_origin: bool = False
    rolling_n_splits: int = 5
    rolling_min_train: int = 50
    rolling_horizon: int = 1

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
    validation_coverage_90: float | None = None
    validation_sharpness_90: float | None = None
    validation_winkler_90: float | None = None
    validation_split_type: str | None = None
    validation_strategy: str | None = None
    validation_n_splits: int | None = None
    test_split_type: str = "out_of_time"


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
    errors: list[dict[str, str]] = field(default_factory=list)


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
        self.errors: list[dict[str, str]] = []
        self._run_result: BenchmarkResult | None = None
        self._tuning_cache: dict[str, dict[str, Any]] = {}
        self._tuning_result_cache: dict[str, Any] = {}

    def load_data(self) -> None:
        """Load the dataset."""
        self.df, self.ds_info = load_dataset(
            self.config.dataset_name,
            n_samples=self.config.n_samples,
            revision=self.config.dataset_revision,
        )
        if self.config.target_column:
            self.target = self.config.target_column
        else:
            self.target = self.ds_info.default_target

    def _get_tuning_result(self, model_name: str, tune_df: pl.DataFrame) -> Any | None:
        """Get tuning result for a model, running tuning if needed."""
        if model_name in self._tuning_result_cache:
            return self._tuning_result_cache[model_name]

        if not self.config.auto_tune:
            return None

        logger.info("Auto-tuning model '%s'", model_name)
        tune_config = TuningConfig(
            target_coverage=self.config.target_coverage,
            n_samples=self.config.tune_samples,
            timeout=self.config.tune_timeout,
            hybrid_validation=self.config.hybrid_validation,
        )

        tuning_result = auto_tune_model(
            model_name=model_name,
            df=tune_df,
            target=self.target,
            horizon=self.config.horizon,
            config=tune_config,
        )

        self._tuning_cache[model_name] = tuning_result.best_params
        self._tuning_result_cache[model_name] = tuning_result
        return tuning_result

    def run_model(self, model_name: str) -> ModelResult:
        """Run a single model benchmark with optional auto-tuning."""
        if model_name not in MODEL_REGISTRY:
            raise ConfigurationError(
                f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        if self.df is None:
            raise DataError("Data not loaded. Call load_data() first.")

        # Determine evaluation data for tuning (use full data head for tuning)
        n_total = len(self.df)
        n_test = int(n_total * self.config.test_size)
        tune_train_df = self.df.head(max(1, n_total - n_test))
        tuning_result = self._get_tuning_result(model_name, tune_train_df)

        if self.config.rolling_origin:
            return self._run_model_rolling_origin(model_name, tuning_result=tuning_result)

        # Temporal train/test split to prevent data leakage
        if n_total < 2:
            raise DataError("Dataset must contain at least 2 rows for benchmark train/test split.")
        if n_test < 1:
            raise DataError(
                "test_size produced an empty test split. "
                "Increase test_size or n_samples so at least one test row is retained."
            )
        if n_test >= n_total:
            raise DataError(
                "test_size produced an empty train split. "
                "Decrease test_size so at least one train row is retained."
            )
        train_df = self.df.head(n_total - n_test)
        test_df = self.df.tail(n_test)

        return self._evaluate_single_split(
            model_name, train_df, test_df, tuning_result=tuning_result
        )

    def _evaluate_single_split(
        self,
        model_name: str,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        tuning_result: Any | None,
    ) -> ModelResult:
        """Evaluate a model on a single train/test split."""
        tuned_params = tuning_result.best_params if tuning_result is not None else {}
        was_tuned = bool(tuned_params)

        if was_tuned:
            logger.info("Using tuned params for '%s': %s", model_name, tuned_params)

        model_cls = MODEL_REGISTRY[model_name]
        benchmark = model_cls(self.config, tuned_params)

        logger.info("Fitting benchmark model '%s'", model_name)
        benchmark.fit(train_df, self.target)

        logger.info("Predicting with benchmark model '%s'", model_name)
        pred = benchmark.predict(test_df)

        n_pred = len(pred.interval(0.9))
        y_true = to_numpy_series(test_df[self.target])[-n_pred:]
        lower_90_s, upper_90_s = pred.interval_bounds(0.9)
        lower_80_s, upper_80_s = pred.interval_bounds(0.8)
        lower_90 = to_numpy_series(lower_90_s)
        upper_90 = to_numpy_series(upper_90_s)
        lower_80 = to_numpy_series(lower_80_s)
        upper_80 = to_numpy_series(upper_80_s)

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
            validation_coverage_90=(
                round(float(getattr(tuning_result, "coverage_90", 0.0)), 4)
                if tuning_result is not None and hasattr(tuning_result, "coverage_90")
                else None
            ),
            validation_sharpness_90=(
                round(float(getattr(tuning_result, "sharpness_90", 0.0)), 4)
                if tuning_result is not None and hasattr(tuning_result, "sharpness_90")
                else None
            ),
            validation_winkler_90=(
                round(float(getattr(tuning_result, "winkler_90", 0.0)), 4)
                if tuning_result is not None and hasattr(tuning_result, "winkler_90")
                else None
            ),
            validation_split_type=(
                str(getattr(tuning_result, "validation_split_type", None))
                if tuning_result is not None and hasattr(tuning_result, "validation_split_type")
                else None
            ),
            validation_strategy=(
                str(getattr(tuning_result, "validation_strategy", None))
                if tuning_result is not None and hasattr(tuning_result, "validation_strategy")
                else None
            ),
            validation_n_splits=(
                int(getattr(tuning_result, "validation_n_splits", 0))
                if tuning_result is not None and hasattr(tuning_result, "validation_n_splits")
                else None
            ),
        )

    def _run_model_rolling_origin(
        self,
        model_name: str,
        tuning_result: Any | None,
    ) -> ModelResult:
        """Evaluate a model using rolling-origin (expanding window) splits."""
        splitter = RollingOriginSplit(
            n_splits=self.config.rolling_n_splits,
            min_train_size=self.config.rolling_min_train,
            horizon=self.config.rolling_horizon,
        )
        if self.df is None:
            raise DataError("Data not loaded. Call load_data() first.")
        splits = splitter.splits(self.df)

        tuned_params = tuning_result.best_params if tuning_result is not None else {}
        was_tuned = bool(tuned_params)

        cov_90s: list[float] = []
        cov_80s: list[float] = []
        wink_90s: list[float] = []
        wink_80s: list[float] = []
        sharp_90s: list[float] = []
        sharp_80s: list[float] = []
        pinballs: list[float] = []
        train_times: list[float] = []
        n_samples_list: list[int] = []

        model_cls = MODEL_REGISTRY[model_name]

        for fold_idx, (train_df, test_df) in enumerate(splits, start=1):
            logger.info(
                "Rolling-origin fold %d/%d for '%s' (train=%d, test=%d)",
                fold_idx,
                len(splits),
                model_name,
                len(train_df),
                len(test_df),
            )

            benchmark = model_cls(self.config, tuned_params)
            start = time.time()
            benchmark.fit(train_df, self.target)
            train_time = time.time() - start

            pred = benchmark.predict(test_df)

            n_pred = len(pred.interval(0.9))
            y_true = to_numpy_series(test_df[self.target])[-n_pred:]
            lower_90_s, upper_90_s = pred.interval_bounds(0.9)
            lower_80_s, upper_80_s = pred.interval_bounds(0.8)
            lower_90 = to_numpy_series(lower_90_s)
            upper_90 = to_numpy_series(upper_90_s)
            lower_80 = to_numpy_series(lower_80_s)
            upper_80 = to_numpy_series(upper_80_s)

            cov_90s.append(coverage_score(y_true, lower_90, upper_90))
            cov_80s.append(coverage_score(y_true, lower_80, upper_80))
            wink_90s.append(winkler_score(y_true, lower_90, upper_90, confidence=0.9))
            wink_80s.append(winkler_score(y_true, lower_80, upper_80, confidence=0.8))
            sharp_90s.append(float(np.mean(upper_90 - lower_90)))
            sharp_80s.append(float(np.mean(upper_80 - lower_80)))
            pinballs.append(float(pinball_loss(y_true, lower_90, 0.1)))
            train_times.append(train_time)
            n_samples_list.append(n_pred)

        return ModelResult(
            model_name=model_name,
            coverage_90=round(float(np.mean(cov_90s)), 4),
            coverage_80=round(float(np.mean(cov_80s)), 4),
            sharpness_90=round(float(np.mean(sharp_90s)), 4),
            sharpness_80=round(float(np.mean(sharp_80s)), 4),
            winkler_90=round(float(np.mean(wink_90s)), 4),
            winkler_80=round(float(np.mean(wink_80s)), 4),
            pinball_loss=round(float(np.mean(pinballs)), 4),
            train_time_sec=round(float(np.sum(train_times)), 3),
            n_samples=int(np.sum(n_samples_list)),
            tuned_params=tuned_params,
            was_tuned=was_tuned,
            test_split_type="rolling_origin",
        )

    def run_all(
        self,
        model_names: list[str] | None = None,
        allow_partial: bool = False,
    ) -> BenchmarkResult:
        """Run all benchmarks for configured dataset.

        Args:
            model_names: List of model names to run. If None, runs all registered.

        Returns:
            BenchmarkResult with all model results
        """
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        self.results = []
        self.errors = []

        for model_name in model_names:
            try:
                result = self.run_model(model_name)
                self.results.append(result)
            except Exception as e:
                error_payload = {"model": model_name, "error": str(e)}
                self.errors.append(error_payload)
                logger.exception("Benchmark model '%s' failed: %s", model_name, e)
                if not allow_partial:
                    raise RuntimeError(f"Benchmark failed for model '{model_name}': {e}") from e

        if not self.results:
            if self.errors:
                raise RuntimeError(
                    f"Benchmark produced no successful model results. Errors: {self.errors}"
                )
            raise RuntimeError("Benchmark produced no model results.")

        self._run_result = BenchmarkResult(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=self.config.dataset_name,
            dataset_domain=self.ds_info.domain if self.ds_info else "Unknown",
            n_samples=self.config.n_samples,
            horizon=self.config.horizon,
            models=self.results,
            errors=self.errors,
        )

        return self._run_result

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        if self._run_result is None:
            return {"metadata": {}, "results": [], "models": []}
        results = [
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
                "validation_coverage_90": r.validation_coverage_90,
                "validation_sharpness_90": r.validation_sharpness_90,
                "validation_winkler_90": r.validation_winkler_90,
                "validation_split_type": r.validation_split_type,
                "validation_strategy": r.validation_strategy,
                "validation_n_splits": r.validation_n_splits,
                "test_split_type": r.test_split_type,
            }
            for r in self._run_result.models
        ]
        return {
            "dataset": self._run_result.dataset_name,
            "metadata": {
                "run_id": self._run_result.run_id,
                "timestamp": self._run_result.timestamp,
                "dataset": self._run_result.dataset_name,
                "domain": self._run_result.dataset_domain,
                "n_samples": self._run_result.n_samples,
                "horizon": self._run_result.horizon,
                "test_size": self.config.test_size,
                "auto_tune": self.config.auto_tune,
                "target_coverage": self.config.target_coverage,
            },
            "errors": self._run_result.errors,
            "results": results,
            "models": results,
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
                    "validation_coverage_90": r.validation_coverage_90,
                    "validation_sharpness_90": r.validation_sharpness_90,
                    "validation_winkler_90": r.validation_winkler_90,
                    "validation_split_type": r.validation_split_type,
                    "validation_strategy": r.validation_strategy,
                    "validation_n_splits": r.validation_n_splits,
                    "test_split_type": r.test_split_type,
                }
            )

        df = pl.DataFrame(rows)
        df.write_csv(path)
