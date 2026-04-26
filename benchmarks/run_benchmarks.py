#!/usr/bin/env python3
"""Consolidated benchmark suite for uncertainty_flow.

Replaces ``scripts/trial_benchmark.py`` and ``scripts/comprehensive_benchmark.py``.

Features
--------
- Uses the library's ``BenchmarkRunner`` / model registry — no duplicated wrappers.
- Registers *baseline* models (linear-regression, ridge, random-forest,
  gradient-boosting, naive-forecast, moving-average) at import time so they
  participate in the same framework.
- Multi-iteration timing with warm-up via ``benchmark_utils.measure_time``.
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

from uncertainty_flow.benchmarking.datasets import load_dataset
from uncertainty_flow.benchmarking.runner import (
    MODEL_REGISTRY,
    BenchmarkConfig,
    ModelResult,
    register_model,
)
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.metrics import coverage_score, pinball_loss, winkler_score
from uncertainty_flow.utils.polars_bridge import to_numpy_series_zero_copy
from uncertainty_flow.wrappers import ConformalRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_utils import TimingStats  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_MODELS = [
    "quantile-forest",
    "conformal-regressor",
    "conformal-forecaster",
    "linear-regression",
    "ridge-regression",
    "random-forest",
    "gradient-boosting",
    "naive-forecast",
    "moving-average",
]

DEFAULT_DATASETS = ["weather", "exchange_rate", "electricity"]


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
    """Register baseline model wrappers (idempotent)."""

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


_register_baselines()


def _evaluate_model(
    config: BenchmarkConfig,
    model_name: str,
    df: pl.DataFrame,
    target: str,
) -> ModelResult:
    """Fit + predict a single model and compute metrics."""
    model_cls = MODEL_REGISTRY[model_name]
    benchmark = model_cls(config, {})

    benchmark.fit(df, target)
    pred = benchmark.predict(df)

    interval_90 = pred.interval(0.9)
    interval_80 = pred.interval(0.8)

    n_pred = len(interval_90)
    y_true = to_numpy_series_zero_copy(df[target])[-n_pred:]
    lower_90 = to_numpy_series_zero_copy(interval_90["lower"])
    upper_90 = to_numpy_series_zero_copy(interval_90["upper"])
    lower_80 = to_numpy_series_zero_copy(interval_80["lower"])
    upper_80 = to_numpy_series_zero_copy(interval_80["upper"])

    cov_90 = coverage_score(y_true, lower_90, upper_90)
    cov_80 = coverage_score(y_true, lower_80, upper_80)
    wink_90 = winkler_score(y_true, lower_90, upper_90, confidence=0.9)
    wink_80 = winkler_score(y_true, lower_80, upper_80, confidence=0.8)
    sharp_90 = float(np.mean(upper_90 - lower_90))
    sharp_80 = float(np.mean(upper_80 - lower_80))
    pinball = pinball_loss(y_true, lower_90, 0.1)

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
    )


def _run_multi_iteration(
    config: BenchmarkConfig,
    model_name: str,
    df: pl.DataFrame,
    target: str,
    n_iterations: int = 5,
    n_warmup: int = 1,
) -> tuple[ModelResult, TimingStats]:
    """Run a model multiple times to get reliable timing statistics."""
    timings: list[float] = []
    last_result: ModelResult | None = None

    for i in range(n_warmup + n_iterations):
        start = time.perf_counter()
        last_result = _evaluate_model(config, model_name, df, target)
        elapsed = time.perf_counter() - start
        if i >= n_warmup:
            timings.append(elapsed)

    assert last_result is not None
    timing_stats = TimingStats(name=model_name, values=timings)
    last_result.train_time_sec = round(timing_stats.mean, 3)
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
    """Run benchmark on a single dataset.

    Returns a dict compatible with the existing report generator.
    """
    if model_names is None:
        model_names = list(DEFAULT_MODELS)

    available = [m for m in model_names if m in MODEL_REGISTRY]
    if not available:
        print(f"  No registered models found for {model_names}")
        return {}

    print(f"
{'=' * 80}")
    print(f"Dataset: {dataset_name} | Samples: {n_samples} | Horizon: {horizon}")
    print(f"Models: {available} | Iterations: {iterations} | Warmup: {warmup}")
    print(f"{'=' * 80}")

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
    timing_rows: list[dict] = []

    for model_name in available:
        print(f"
  [{model_name}]")
        try:
            model_result, timing = _run_multi_iteration(
                config, model_name, df, target, n_iterations=iterations, n_warmup=warmup
            )
            print(f"    Coverage@90%: {model_result.coverage_90:.4f}")
            print(f"    Winkler@90%:  {model_result.winkler_90:.4f}")
            print(f"    Sharpness@90%: {model_result.sharpness_90:.4f}")
            print(f"    Pinball:      {model_result.pinball_loss:.4f}")
            print(f"    Timing: {timing.summary()}")

            results.append(
                {
                    "model": model_result.model_name,
                    "coverage_90": model_result.coverage_90,
                    "coverage_80": model_result.coverage_80,
                    "sharpness_90": model_result.sharpness_90,
                    "sharpness_80": model_result.sharpness_80,
                    "winkler_90": model_result.winkler_90,
                    "winkler_80": model_result.winkler_80,
                    "pinball_loss": model_result.pinball_loss,
                    "train_time_sec": model_result.train_time_sec,
                    "n_samples": model_result.n_samples,
                    **timing.to_dict(),
                }
            )
            timing_rows.append(
                {
                    "model": model_name,
                    **timing.to_dict(),
                }
            )
        except Exception as e:
            print(f"    ERROR: {e}")

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

        print(f"
  Saved: {base}_{dataset_name}.json / .csv")

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
    """Run benchmarks across multiple datasets."""
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
        print(f"
Combined results saved to: {path}")

    _print_summary(all_results)
    return all_results


def _print_summary(all_results: dict[str, dict]) -> None:
    print(f"
{'=' * 80}")
    print("CONSOLIDATED BENCHMARK SUMMARY")
    print(f"{'=' * 80}")

    for ds, payload in all_results.items():
        results = payload.get("results", [])
        if not results:
            continue

        print(f"
--- {ds} ---")
        print(
            f"  {'Model':<22} {'Cov@90':>8} "
            f"{'Wink@90':>10} {'Sharp@90':>10} "
            f"{'Time(s)':>10} {'±std':>10}"
        )
        print(f"  {'-' * 70}")
        for r in sorted(results, key=lambda x: x.get("winkler_90", float("inf"))):
            print(
                f"  {r['model']:<22} {r.get('coverage_90', 0):>8.4f} "
                f"{r.get('winkler_90', 0):>10.4f} {r.get('sharpness_90', 0):>10.4f} "
                f"{r.get('mean', r.get('train_time_sec', 0)):>10.4f} "
                f"{r.get('std', 0):>10.4f}"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Consolidated benchmark suite for uncertainty_flow",
    )
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Single dataset")
    parser.add_argument("--all-datasets", action="store_true", help="Run on all default datasets")
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

    args = parser.parse_args()
    models = args.models.split(",") if args.models else None

    if args.all_datasets:
        run_all(
            n_samples=args.n_samples,
            horizon=args.horizon,
            iterations=args.iterations,
            warmup=args.warmup,
            auto_tune=args.auto_tune,
            output_prefix=args.output,
            model_names=models,
        )
    elif args.dataset:
        run_benchmark(
            dataset_name=args.dataset,
            model_names=models,
            n_samples=args.n_samples,
            horizon=args.horizon,
            iterations=args.iterations,
            warmup=args.warmup,
            auto_tune=args.auto_tune,
            output_prefix=args.output,
        )
    else:
        parser.error("Provide --dataset <name> or --all-datasets")


if __name__ == "__main__":
    main()
