#!/usr/bin/env python3
"""Comprehensive benchmark comparing uncertainty_flow models with conventional baselines.

This script extends the uncertainty_flow benchmarking framework to include:
- Conventional regression models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- Simple time series baselines (Naive, Seasonal Naive, Moving Average)
- Quantile regression baselines

Run with: uv run python scripts/comprehensive_benchmark.py
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from uncertainty_flow.benchmarking.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    register_model,
)
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.wrappers import ConformalRegressor

# ============================================================================
# Conventional Baseline Models
# ============================================================================


@register_model("linear-regression")
class LinearRegressionBenchmark:
    """Benchmark wrapper for Linear Regression with conformalized intervals."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.model: ConformalRegressor | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        calib = self.tuned_params.get("calibration_size", 0.2)

        self.model = ConformalRegressor(
            base_model=LinearRegression(),
            calibration_size=calib,
            auto_tune=False,
            random_state=self.config.random_state,
        )
        start = time.time()
        self.model.fit(df, target=target)
        self.train_time = time.time() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(df)


@register_model("ridge-regression")
class RidgeRegressionBenchmark:
    """Benchmark wrapper for Ridge Regression with conformalized intervals."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.model: ConformalRegressor | None = None

    def fit(self, df: pl.DataFrame, target: str) -> None:
        calib = self.tuned_params.get("calibration_size", 0.2)
        alpha = self.tuned_params.get("alpha", 1.0)

        self.model = ConformalRegressor(
            base_model=Ridge(alpha=alpha),
            calibration_size=calib,
            auto_tune=False,
            random_state=self.config.random_state,
        )
        start = time.time()
        self.model.fit(df, target=target)
        self.train_time = time.time() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(df)


@register_model("random-forest")
class RandomForestBenchmark:
    """Benchmark wrapper for Random Forest with conformalized intervals."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.model: ConformalRegressor | None = None

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
        start = time.time()
        self.model.fit(df, target=target)
        self.train_time = time.time() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(df)


@register_model("gradient-boosting")
class GradientBoostingBenchmark:
    """Benchmark wrapper for Gradient Boosting with conformalized intervals."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
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
            raise RuntimeError("Model not fitted")
        return self.model.predict(df)


@register_model("naive-forecast")
class NaiveForecastBenchmark:
    """Naive forecast: predicts last observed value with historical error-based intervals."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.last_value: float | None = None
        self.residual_std: float | None = None
        self.train_time: float = 0.0

    def fit(self, df: pl.DataFrame, target: str) -> None:
        start = time.time()
        y = df[target].to_numpy()

        # Use last value as prediction
        self.last_value = y[-1]

        # Estimate residual standard deviation from historical differences
        if len(y) > 1:
            diffs = np.diff(y)
            self.residual_std = np.std(diffs) * np.sqrt(self.config.horizon)
        else:
            self.residual_std = np.std(y)

        self.train_time = time.time() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.last_value is None or self.residual_std is None:
            raise RuntimeError("Model not fitted")

        n = len(df)
        # Create prediction intervals based on normal distribution assumption
        # 90% interval: ±1.645 std, 80% interval: ±1.28 std
        z_90 = 1.645
        z_80 = 1.28

        lower_90 = np.full(n, self.last_value - z_90 * self.residual_std)
        upper_90 = np.full(n, self.last_value + z_90 * self.residual_std)
        lower_80 = np.full(n, self.last_value - z_80 * self.residual_std)
        upper_80 = np.full(n, self.last_value + z_80 * self.residual_std)

        # Create a simple DistributionPrediction-like object
        return SimpleDistributionPrediction(
            lower_90=lower_90,
            upper_90=upper_90,
            lower_80=lower_80,
            upper_80=upper_80,
        )


@register_model("moving-average")
class MovingAverageBenchmark:
    """Moving average forecast with historical error-based intervals."""

    def __init__(self, config: BenchmarkConfig, tuned_params: dict | None = None):
        self.config = config
        self.tuned_params = tuned_params or {}
        self.ma_value: float | None = None
        self.residual_std: float | None = None
        self.train_time: float = 0.0
        self.window = self.tuned_params.get("window", 5)

    def fit(self, df: pl.DataFrame, target: str) -> None:
        start = time.time()
        y = df[target].to_numpy()

        # Use moving average as prediction
        if len(y) >= self.window:
            self.ma_value = np.mean(y[-self.window :])
        else:
            self.ma_value = np.mean(y)

        # Estimate residual standard deviation
        if len(y) > self.window:
            residuals = y[self.window :] - np.array(
                [np.mean(y[i - self.window : i]) for i in range(self.window, len(y))]
            )
            self.residual_std = np.std(residuals)
        else:
            self.residual_std = np.std(y)

        self.train_time = time.time() - start

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        if self.ma_value is None or self.residual_std is None:
            raise RuntimeError("Model not fitted")

        n = len(df)
        z_90 = 1.645
        z_80 = 1.28

        lower_90 = np.full(n, self.ma_value - z_90 * self.residual_std)
        upper_90 = np.full(n, self.ma_value + z_90 * self.residual_std)
        lower_80 = np.full(n, self.ma_value - z_80 * self.residual_std)
        upper_80 = np.full(n, self.ma_value + z_80 * self.residual_std)

        return SimpleDistributionPrediction(
            lower_90=lower_90,
            upper_90=upper_90,
            lower_80=lower_80,
            upper_80=upper_80,
        )


@dataclass
class SimpleDistributionPrediction:
    """Simple distribution prediction wrapper for baseline models."""

    lower_90: np.ndarray
    upper_90: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray
    _targets: list[str] = field(default_factory=lambda: ["prediction"])

    def interval(self, confidence: float) -> pl.DataFrame:
        """Return prediction interval as DataFrame."""
        if confidence == 0.9:
            return pl.DataFrame({"lower": self.lower_90, "upper": self.upper_90})
        elif confidence == 0.8:
            return pl.DataFrame({"lower": self.lower_80, "upper": self.upper_80})
        else:
            raise ValueError(f"Unsupported confidence level: {confidence}")


# ============================================================================
# Benchmark Runner
# ============================================================================


def run_comprehensive_benchmark(
    dataset_name: str,
    n_samples: int = 500,
    horizon: int = 3,
    auto_tune: bool = False,
    output_prefix: str | None = None,
) -> dict:
    """Run comprehensive benchmark on a dataset.

    Args:
        dataset_name: Name of the dataset to benchmark
        n_samples: Number of samples to use
        horizon: Forecast horizon
        auto_tune: Whether to enable auto-tuning
        output_prefix: Prefix for output files

    Returns:
        Dictionary with benchmark results
    """
    print("=" * 80)
    print(f"Comprehensive Benchmark: {dataset_name}")
    print("=" * 80)
    print(f"  Samples: {n_samples}")
    print(f"  Horizon: {horizon}")
    print(f"  Auto-tuning: {auto_tune}")
    print()

    config = BenchmarkConfig(
        dataset_name=dataset_name,
        n_samples=n_samples,
        horizon=horizon,
        auto_tune=auto_tune,
        random_state=42,
    )

    runner = BenchmarkRunner(config)
    runner.load_data()

    # Run all registered models
    all_models = [
        # Uncertainty flow models
        "quantile-forest",
        "conformal-regressor",
        "conformal-forecaster",
        # Conventional regression baselines
        "linear-regression",
        "ridge-regression",
        "random-forest",
        "gradient-boosting",
        # Simple time series baselines
        "naive-forecast",
        "moving-average",
    ]

    result = runner.run_all(model_names=all_models)

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    for model_result in result.models:
        print(f"\n  {model_result.model_name}:")
        print(f"    Coverage @ 90%: {model_result.coverage_90:.4f}")
        print(f"    Coverage @ 80%: {model_result.coverage_80:.4f}")
        print(f"    Sharpness @ 90%: {model_result.sharpness_90:.4f}")
        print(f"    Sharpness @ 80%: {model_result.sharpness_80:.4f}")
        print(f"    Winkler @ 90%: {model_result.winkler_90:.4f}")
        print(f"    Winkler @ 80%: {model_result.winkler_80:.4f}")
        print(f"    Pinball Loss: {model_result.pinball_loss:.4f}")
        print(f"    Train time: {model_result.train_time_sec:.3f}s")
        if model_result.was_tuned:
            print(f"    Tuned params: {model_result.tuned_params}")

    # Save results
    if output_prefix:
        output_path = Path(output_prefix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runner.save_json(f"{output_prefix}.json")
        runner.save_csv(f"{output_prefix}.csv")
        print(f"\nJSON results saved to: {output_prefix}.json")
        print(f"CSV results saved to: {output_prefix}.csv")

    return runner.to_dict()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark with conventional baselines"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="weather",
        help="Dataset name (default: weather)",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=500,
        help="Number of samples (default: 500)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forecast horizon (default: 3)",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Enable auto-tuning",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file prefix",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run on all default datasets (weather, electricity, exchange_rate)",
    )

    args = parser.parse_args()

    if args.all_datasets:
        datasets = ["weather", "electricity", "exchange_rate"]
        all_results = {}

        for ds in datasets:
            output_prefix = f"{args.output}_{ds}" if args.output else None
            result = run_comprehensive_benchmark(
                dataset_name=ds,
                n_samples=args.n_samples,
                horizon=args.horizon,
                auto_tune=args.auto_tune,
                output_prefix=output_prefix,
            )
            all_results[ds] = result

        # Save combined results
        if args.output:
            import json

            with open(f"{args.output}_all.json", "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nCombined results saved to: {args.output}_all.json")

    else:
        run_comprehensive_benchmark(
            dataset_name=args.dataset,
            n_samples=args.n_samples,
            horizon=args.horizon,
            auto_tune=args.auto_tune,
            output_prefix=args.output,
        )


if __name__ == "__main__":
    main()
