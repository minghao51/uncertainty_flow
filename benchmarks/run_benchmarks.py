#!/usr/bin/env python3
"""Consolidated benchmark suite for uncertainty_flow.

Features
--------
- Uses the library's ``BenchmarkRunner`` / model registry.
- Registers *all* library models + conventional baselines.
- Multi-iteration timing with warm-up via ``benchmark_utils.measure_time``.
- Expanded metrics: coverage, sharpness, Winkler, pinball, CRPS, MAE, RMSE,
  calibration error, and memory delta.
- Outputs JSON, CSV, and a console report.

Usage
-----
    uv run python benchmarks/run_benchmarks.py --all-datasets
    uv run python benchmarks/run_benchmarks.py -d weather -n 500 --iterations 3
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from uncertainty_flow.benchmarking.datasets import load_dataset, load_local_dataset
from uncertainty_flow.benchmarking.runner import (
    MODEL_REGISTRY,
    BenchmarkConfig,
    register_model,
)
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.metrics import (
    calibration_error,
    coverage_score,
    crps_score,
    mae_score,
    pinball_loss,
    rmse_score,
    winkler_score,
)
from uncertainty_flow.utils.polars_bridge import to_numpy_series
from uncertainty_flow.wrappers import ConformalRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_utils import TimingStats  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

UF_MODELS = [
    "quantile-forest",
    "conformal-regressor",
    "conformal-forecaster",
    "deep-quantile",
    "deep-quantile-torch",
    "transformer-forecaster",
    "bayesian-quantile",
]

BASELINE_MODELS = [
    "linear-regression",
    "ridge-regression",
    "random-forest",
    "gradient-boosting",
    "naive-forecast",
    "moving-average",
]

DEFAULT_MODELS = UF_MODELS + BASELINE_MODELS

DEFAULT_DATASETS = ["weather", "exchange_rate", "electricity"]

DATASET_GROUPS: dict[str, list[str]] = {
    "ts": [
        "weather",
        "exchange_rate",
        "electricity",
        "traffic",
        "illness",
        "beijingpm25_local",
        "fraud",
    ],
    "tabular": [
        "concrete",
        "wine_quality",
        "energy_efficiency",
    ],
    "synthetic": [
        "synthetic_heteroscedastic",
        "synthetic_regime_switching",
        "synthetic_heavy_tailed",
        "synthetic_multivariate",
        "financial_volatility",
    ],
}

EXPANDED_DATASETS = DATASET_GROUPS["ts"] + DATASET_GROUPS["tabular"] + DATASET_GROUPS["synthetic"]


@dataclass
class SimpleDistributionPrediction:
    lower_90: np.ndarray
    upper_90: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray
    _targets: list[str] = field(default_factory=lambda: ["prediction"])

    def interval(self, confidence: float) -> pl.DataFrame:
        if confidence == 0.9:
            return pl.DataFrame({"lower": self.lower_90, "upper": self.upper_90})
        if confidence == 0.8:
            return pl.DataFrame({"lower": self.lower_80, "upper": self.upper_80})
        raise ValueError(f"Unsupported confidence level: {confidence}")


def _register_baselines() -> None:
    if "linear-regression" in MODEL_REGISTRY:
        return

    @register_model("linear-regression")
    class _LinearRegressionBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model: ConformalRegressor | None = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            calib = self.tuned_params.get("calibration_size", 0.2)
            self.model = ConformalRegressor(
                base_model=LinearRegression(),
                calibration_size=calib,
                auto_tune=False,
                random_state=self.config.random_state,
            )
            start = time.perf_counter()
            self.model.fit(df, target=target)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)

    @register_model("ridge-regression")
    class _RidgeRegressionBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model: ConformalRegressor | None = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            calib = self.tuned_params.get("calibration_size", 0.2)
            alpha = self.tuned_params.get("alpha", 1.0)
            self.model = ConformalRegressor(
                base_model=Ridge(alpha=alpha),
                calibration_size=calib,
                auto_tune=False,
                random_state=self.config.random_state,
            )
            start = time.perf_counter()
            self.model.fit(df, target=target)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)

    @register_model("random-forest")
    class _RandomForestBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model: ConformalRegressor | None = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            n_est = self.tuned_params.get("n_estimators", self.config.n_estimators)
            calib = self.tuned_params.get("calibration_size", 0.2)
            self.model = ConformalRegressor(
                base_model=RandomForestRegressor(
                    n_estimators=n_est,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                ),
                calibration_size=calib,
                auto_tune=False,
                random_state=self.config.random_state,
            )
            start = time.perf_counter()
            self.model.fit(df, target=target)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)

    @register_model("gradient-boosting")
    class _GradientBoostingBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model: ConformalRegressor | None = None
            self.train_time: float = 0.0

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
            start = time.perf_counter()
            self.model.fit(df, target=target)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)

    @register_model("naive-forecast")
    class _NaiveForecastBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.last_value: float | None = None
            self.residual_std: float | None = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            start = time.perf_counter()
            y = df[target].to_numpy()
            self.last_value = y[-1]
            if len(y) > 1:
                diffs = np.diff(y)
                self.residual_std = np.std(diffs) * np.sqrt(self.config.horizon)
            else:
                self.residual_std = np.std(y)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> SimpleDistributionPrediction:
            if self.last_value is None or self.residual_std is None:
                raise RuntimeError("Model not fitted")
            n = len(df)
            z_90, z_80 = 1.645, 1.28
            return SimpleDistributionPrediction(
                lower_90=np.full(n, self.last_value - z_90 * self.residual_std),
                upper_90=np.full(n, self.last_value + z_90 * self.residual_std),
                lower_80=np.full(n, self.last_value - z_80 * self.residual_std),
                upper_80=np.full(n, self.last_value + z_80 * self.residual_std),
            )

    @register_model("moving-average")
    class _MovingAverageBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.ma_value: float | None = None
            self.residual_std: float | None = None
            self.train_time: float = 0.0
            self.window = self.tuned_params.get("window", 5)

        def fit(self, df: pl.DataFrame, target: str) -> None:
            start = time.perf_counter()
            y = df[target].to_numpy()
            if len(y) >= self.window:
                self.ma_value = float(np.mean(y[-self.window :]))
            else:
                self.ma_value = float(np.mean(y))
            if len(y) > self.window:
                residuals = y[self.window :] - np.array(
                    [np.mean(y[i - self.window : i]) for i in range(self.window, len(y))]
                )
                self.residual_std = float(np.std(residuals))
            else:
                self.residual_std = float(np.std(y))
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> SimpleDistributionPrediction:
            if self.ma_value is None or self.residual_std is None:
                raise RuntimeError("Model not fitted")
            n = len(df)
            z_90, z_80 = 1.645, 1.28
            return SimpleDistributionPrediction(
                lower_90=np.full(n, self.ma_value - z_90 * self.residual_std),
                upper_90=np.full(n, self.ma_value + z_90 * self.residual_std),
                lower_80=np.full(n, self.ma_value - z_80 * self.residual_std),
                upper_80=np.full(n, self.ma_value + z_80 * self.residual_std),
            )


def _register_optional_models() -> None:
    if "deep-quantile" in MODEL_REGISTRY:
        return

    @register_model("deep-quantile")
    class _DeepQuantileBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            from uncertainty_flow.models import DeepQuantileNet

            self.model = DeepQuantileNet(
                hidden_layer_sizes=self.tuned_params.get("hidden_layer_sizes", (64, 32)),
                trunk_max_iter=self.tuned_params.get("trunk_max_iter", 300),
                random_state=self.config.random_state,
            )
            start = time.perf_counter()
            self.model.fit(df, target=target)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)

    @register_model("deep-quantile-torch")
    class _DeepQuantileTorchBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            try:
                from uncertainty_flow.models import DeepQuantileNetTorch
            except ImportError:
                raise ImportError(
                    "torch required for deep-quantile-torch. Install: uv sync --extra opinion"
                )

            self.model = DeepQuantileNetTorch(
                hidden_layer_sizes=self.tuned_params.get("hidden_layer_sizes", (64, 32)),
                epochs=self.tuned_params.get("epochs", 100),
                learning_rate=self.tuned_params.get("learning_rate", 0.001),
                device="cpu",
                random_state=self.config.random_state,
            )
            start = time.perf_counter()
            self.model.fit(df, target=target)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)

    @register_model("transformer-forecaster")
    class _TransformerForecasterBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            try:
                from uncertainty_flow.models import TransformerForecaster
            except ImportError:
                raise ImportError("chronos-forecasting required. Install: uv sync --extra opinion")

            model_name = self.tuned_params.get("chronos_model", "chronos-bolt-tiny")
            self.model = TransformerForecaster(
                target=target,
                horizon=self.config.horizon,
                model_name=model_name,
                calibration_size=self.tuned_params.get("calibration_size", 0.2),
                auto_tune=False,
                device="cpu",
                random_state=self.config.random_state,
            )
            start = time.perf_counter()
            self.model.fit(df)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)

    @register_model("bayesian-quantile")
    class _BayesianQuantileBenchmark:
        def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
            self.config = config
            self.tuned_params = tuned_params or {}
            self.model = None
            self.train_time: float = 0.0

        def fit(self, df: pl.DataFrame, target: str) -> None:
            try:
                from uncertainty_flow.bayesian import BayesianQuantileRegressor
            except ImportError:
                raise ImportError("numpyro is required. Install with: uv sync --extra opinion")

            self.model = BayesianQuantileRegressor(
                n_warmup=self.tuned_params.get("n_warmup", 1000),
                n_samples=self.tuned_params.get("n_samples", 2000),
                prior_width=self.tuned_params.get("prior_width", 10.0),
                random_state=self.config.random_state,
            )
            start = time.perf_counter()
            self.model.fit(df, target=target)
            self.train_time = time.perf_counter() - start

        def predict(self, df: pl.DataFrame) -> DistributionPrediction:
            if self.model is None:
                raise RuntimeError("Model not fitted")
            return self.model.predict(df)


_register_baselines()
_register_optional_models()


def _get_point_prediction(pred) -> np.ndarray:
    """Extract median/point prediction from any prediction type."""
    if isinstance(pred, SimpleDistributionPrediction):
        return (pred.lower_90 + pred.upper_90) / 2.0
    mean_val = pred.median()
    if isinstance(mean_val, pl.DataFrame):
        return mean_val.to_numpy().flatten()
    return to_numpy_series(mean_val)


def _evaluate_model(
    config: BenchmarkConfig,
    model_name: str,
    df: pl.DataFrame,
    target: str,
) -> dict:
    model_cls = MODEL_REGISTRY[model_name]
    benchmark = model_cls(config, {})

    mem_before = _measure_memory_mb()
    benchmark.fit(df, target)
    pred = benchmark.predict(df)
    mem_after = _measure_memory_mb()

    interval_90 = pred.interval(0.9)
    interval_80 = pred.interval(0.8)

    n_pred = len(interval_90)
    y_true = to_numpy_series(df[target])[-n_pred:]
    lower_90 = to_numpy_series(interval_90["lower"])
    upper_90 = to_numpy_series(interval_90["upper"])
    lower_80 = to_numpy_series(interval_80["lower"])
    upper_80 = to_numpy_series(interval_80["upper"])

    y_pred = _get_point_prediction(pred)
    if len(y_pred) > n_pred:
        y_pred = y_pred[-n_pred:]
    elif len(y_pred) < n_pred:
        y_pred = np.resize(y_pred, n_pred)

    cov_90 = coverage_score(y_true, lower_90, upper_90)
    cov_80 = coverage_score(y_true, lower_80, upper_80)
    wink_90 = winkler_score(y_true, lower_90, upper_90, confidence=0.9)
    wink_80 = winkler_score(y_true, lower_80, upper_80, confidence=0.8)
    sharp_90 = float(np.mean(upper_90 - lower_90))
    sharp_80 = float(np.mean(upper_80 - lower_80))
    pinball = pinball_loss(y_true, lower_90, 0.1)
    crps = crps_score(y_true, lower_90, upper_90, confidence=0.9)
    mae = mae_score(y_true, y_pred)
    rmse = rmse_score(y_true, y_pred)
    cal_error = calibration_error(y_true, lower_90, upper_90, nominal_coverage=0.9)

    return {
        "model": model_name,
        "coverage_90": round(cov_90, 4),
        "coverage_80": round(cov_80, 4),
        "sharpness_90": round(sharp_90, 4),
        "sharpness_80": round(sharp_80, 4),
        "winkler_90": round(wink_90, 4),
        "winkler_80": round(wink_80, 4),
        "pinball_loss": round(float(pinball), 4),
        "crps": round(float(crps), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "calibration_error": round(float(cal_error), 4),
        "train_time_sec": round(benchmark.train_time, 3),
        "memory_delta_mb": round(max(0, mem_after - mem_before), 2),
        "n_samples": n_pred,
    }


def _run_multi_iteration(
    config: BenchmarkConfig,
    model_name: str,
    df: pl.DataFrame,
    target: str,
    n_iterations: int = 3,
    n_warmup: int = 1,
) -> tuple[dict, TimingStats]:
    timings: list[float] = []
    last_result: dict | None = None

    for i in range(n_warmup + n_iterations):
        start = time.perf_counter()
        last_result = _evaluate_model(config, model_name, df, target)
        elapsed = time.perf_counter() - start
        if i >= n_warmup:
            timings.append(elapsed)

    assert last_result is not None
    timing_stats = TimingStats(name=model_name, values=timings)
    last_result["train_time_sec"] = round(timing_stats.mean, 3)
    last_result["timing_mean"] = round(timing_stats.mean, 6)
    last_result["timing_std"] = round(timing_stats.std, 6)
    last_result["n_runs"] = len(timings)
    return last_result, timing_stats


def run_benchmark(
    dataset_name: str,
    model_names: list[str] | None = None,
    n_samples: int = 500,
    horizon: int = 3,
    iterations: int = 3,
    warmup: int = 1,
    auto_tune: bool = False,
    output_prefix: str | None = None,
) -> dict:
    if model_names is None:
        model_names = list(DEFAULT_MODELS)

    available = []
    skipped = []
    for m in model_names:
        if m in MODEL_REGISTRY:
            available.append(m)
        else:
            skipped.append(m)

    if skipped:
        print(f"  Skipped (not registered): {skipped}")

    if not available:
        print(f"  No registered models found for {model_names}")
        return {}

    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name} | Samples: {n_samples} | Horizon: {horizon}")
    print(f"Models: {available} | Iterations: {iterations} | Warmup: {warmup}")
    print(f"{'=' * 80}")

    try:
        df, ds_info = load_local_dataset(dataset_name, n_samples=n_samples)
    except Exception:
        df, ds_info = load_dataset(dataset_name, n_samples=n_samples)
    target = ds_info.default_target
    print(f"  Loaded {len(df):,} rows, target={target}")

    config = BenchmarkConfig(
        dataset_name=dataset_name,
        n_samples=n_samples,
        horizon=horizon,
        auto_tune=auto_tune,
        random_state=42,
    )

    results: list[dict] = []
    errors: list[dict[str, str]] = []

    for model_name in available:
        print(f"\n  [{model_name}]")
        try:
            model_result, timing = _run_multi_iteration(
                config, model_name, df, target, n_iterations=iterations, n_warmup=warmup
            )
            print(f"    Coverage@90%: {model_result['coverage_90']:.4f}")
            print(f"    Winkler@90%:  {model_result['winkler_90']:.4f}")
            print(f"    CRPS:         {model_result['crps']:.4f}")
            print(f"    MAE:          {model_result['mae']:.4f}")
            print(f"    RMSE:         {model_result['rmse']:.4f}")
            print(f"    Cal.Error:    {model_result['calibration_error']:.4f}")
            print(f"    Timing: {timing.summary()}")

            results.append(model_result)
        except ImportError as e:
            print(f"    SKIP (missing dep): {e}")
            errors.append(
                {
                    "model": model_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append(
                {
                    "model": model_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )

    payload = {
        "metadata": {
            "dataset": dataset_name,
            "domain": ds_info.domain,
            "n_samples": n_samples,
            "horizon": horizon,
            "iterations": iterations,
            "warmup": warmup,
            "auto_tune": auto_tune,
        },
        "results": results,
        "errors": errors,
    }

    if output_prefix:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        base = RESULTS_DIR / output_prefix

        with open(f"{base}_{dataset_name}.json", "w") as f:
            json.dump(payload, f, indent=2)

        if results:
            rows = []
            for r in results:
                rows.append({"dataset": dataset_name, **r})
            pl.DataFrame(rows).write_csv(f"{base}_{dataset_name}.csv")

        print(f"\n  Saved: {base}_{dataset_name}.json / .csv")

    return payload


def run_all(
    datasets: list[str] | None = None,
    model_names: list[str] | None = None,
    n_samples: int = 500,
    horizon: int = 3,
    iterations: int = 3,
    warmup: int = 1,
    auto_tune: bool = False,
    output_prefix: str | None = "consolidated",
) -> dict[str, dict]:
    if datasets is None:
        datasets = DEFAULT_DATASETS

    all_results: dict[str, dict] = {}
    for ds in datasets:
        all_results[ds] = run_benchmark(
            dataset_name=ds,
            model_names=model_names,
            n_samples=n_samples,
            horizon=horizon,
            iterations=iterations,
            warmup=warmup,
            auto_tune=auto_tune,
            output_prefix=output_prefix,
        )

    if output_prefix:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{output_prefix}_all.json"
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to: {path}")

    _print_summary(all_results)
    return all_results


def _print_summary(all_results: dict[str, dict]) -> None:
    print(f"\n{'=' * 80}")
    print("CONSOLIDATED BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    for ds, payload in all_results.items():
        results = payload.get("results", [])
        if not results:
            continue

        print(f"\n--- {ds} ---")
        print(
            f"  {'Model':<24} {'Cov@90':>7} "
            f"{'Wink@90':>9} {'CRPS':>9} {'MAE':>9} "
            f"{'CalErr':>7} {'Time(s)':>9} {'±std':>9}"
        )
        print(f"  {'-' * 83}")
        for r in sorted(results, key=lambda x: x.get("winkler_90", float("inf"))):
            print(
                f"  {r['model']:<24} {r.get('coverage_90', 0):>7.4f} "
                f"{r.get('winkler_90', 0):>9.4f} {r.get('crps', 0):>9.4f} "
                f"{r.get('mae', 0):>9.4f} {r.get('calibration_error', 0):>7.4f} "
                f"{r.get('timing_mean', r.get('train_time_sec', 0)):>9.4f} "
                f"{r.get('timing_std', 0):>9.4f}"
            )


def _measure_memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Consolidated benchmark suite for uncertainty_flow",
    )
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Single dataset")
    parser.add_argument("--all-datasets", action="store_true", help="Run on all default datasets")
    parser.add_argument(
        "--dataset-group",
        "-g",
        type=str,
        default=None,
        choices=list(DATASET_GROUPS.keys()) + ["expanded"],
        help="Run a predefined group of datasets (ts, tabular, synthetic, expanded)",
    )
    parser.add_argument("--n-samples", "-n", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--iterations", "-i", type=int, default=3, help="Timed iterations")
    parser.add_argument("--warmup", "-w", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--auto-tune", action="store_true")
    parser.add_argument("--output", "-o", type=str, default="consolidated")
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        default=None,
        help="Comma-separated model names (default: all registered)",
    )
    parser.add_argument(
        "--uf-only",
        action="store_true",
        help="Run only uncertainty_flow models (no baselines)",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run only baseline models",
    )

    args = parser.parse_args()

    if args.models:
        model_names = args.models.split(",")
    elif args.uf_only:
        model_names = UF_MODELS
    elif args.baselines_only:
        model_names = BASELINE_MODELS
    else:
        model_names = None

    if args.all_datasets:
        datasets = EXPANDED_DATASETS
    elif args.dataset_group:
        if args.dataset_group == "expanded":
            datasets = EXPANDED_DATASETS
        else:
            datasets = DATASET_GROUPS[args.dataset_group]
    else:
        datasets = None

    if datasets is not None:
        run_all(
            datasets=datasets,
            n_samples=args.n_samples,
            horizon=args.horizon,
            iterations=args.iterations,
            warmup=args.warmup,
            auto_tune=args.auto_tune,
            output_prefix=args.output,
            model_names=model_names,
        )
    elif args.dataset:
        run_benchmark(
            dataset_name=args.dataset,
            model_names=model_names,
            n_samples=args.n_samples,
            horizon=args.horizon,
            iterations=args.iterations,
            warmup=args.warmup,
            auto_tune=args.auto_tune,
            output_prefix=args.output,
        )
    else:
        parser.error("Provide --dataset <name>, --all-datasets, or --dataset-group <group>")


if __name__ == "__main__":
    main()
