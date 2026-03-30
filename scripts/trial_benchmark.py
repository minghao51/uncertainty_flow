#!/usr/bin/env python3
"""Trial benchmark for uncertainty_flow models on sample datasets.

Run with: uv run python scripts/trial_benchmark.py
"""

import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.metrics import coverage_score, winkler_score
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    n_samples: int
    train_time_sec: float
    coverage_90: float
    sharpness_90: float
    winkler_90: float


def load_dataset(name: str) -> tuple[pl.DataFrame, str]:
    """Load dataset and return (dataframe, target_column)."""
    data_dir = Path(__file__).parent.parent / "data"

    targets = {
        "weather": "OT",
        "exchange_rate": "OT",
        "electricity": "OT",
    }

    df = pl.read_parquet(data_dir / f"{name}.parquet")
    return df, targets[name]


def benchmark_quantile_forest(
    df: pl.DataFrame, target: str, n_samples: int = 1000
) -> BenchmarkResult:
    """Benchmark QuantileForestForecaster."""
    df_sub = df.head(n_samples)
    horizon = 3

    start = time.time()
    model = QuantileForestForecaster(
        targets=target,
        horizon=horizon,
        n_estimators=30,
        random_state=42,
    )
    model.fit(df_sub)
    pred = model.predict(df_sub)
    train_time = time.time() - start

    # Align predictions with true values
    interval_df = pred.interval(0.9)
    n_pred = len(interval_df)
    y_true = df_sub[target].to_numpy()[-n_pred:]
    lower = interval_df["lower"].to_numpy()
    upper = interval_df["upper"].to_numpy()

    cov = coverage_score(y_true, lower, upper)
    wink = winkler_score(y_true, lower, upper, confidence=0.9)
    sharpness = (upper - lower).mean()

    return BenchmarkResult(
        dataset="",
        model="QuantileForestForecaster",
        n_samples=n_samples,
        train_time_sec=round(train_time, 3),
        coverage_90=round(cov, 4),
        sharpness_90=round(float(sharpness), 4),
        winkler_90=round(wink, 4),
    )


def benchmark_conformal_regressor(
    df: pl.DataFrame, target: str, n_samples: int = 1000
) -> BenchmarkResult:
    """Benchmark ConformalRegressor."""
    df_sub = df.head(n_samples)

    start = time.time()
    model = ConformalRegressor(
        base_model=GradientBoostingRegressor(n_estimators=20, random_state=42),
        random_state=42,
    )
    model.fit(df_sub, target=target)
    pred = model.predict(df_sub)
    train_time = time.time() - start

    interval_df = pred.interval(0.9)
    n_pred = len(interval_df)
    y_true = df_sub[target].to_numpy()[-n_pred:]
    lower = interval_df["lower"].to_numpy()
    upper = interval_df["upper"].to_numpy()

    cov = coverage_score(y_true, lower, upper)
    wink = winkler_score(y_true, lower, upper, confidence=0.9)
    sharpness = (upper - lower).mean()

    return BenchmarkResult(
        dataset="",
        model="ConformalRegressor",
        n_samples=n_samples,
        train_time_sec=round(train_time, 3),
        coverage_90=round(cov, 4),
        sharpness_90=round(float(sharpness), 4),
        winkler_90=round(wink, 4),
    )


def benchmark_conformal_forecaster(
    df: pl.DataFrame, target: str, n_samples: int = 1000
) -> BenchmarkResult:
    """Benchmark ConformalForecaster."""
    df_sub = df.head(n_samples)
    horizon = 3

    start = time.time()
    model = ConformalForecaster(
        base_model=GradientBoostingRegressor(n_estimators=20, random_state=42),
        horizon=horizon,
        targets=target,
        lags=2,
        random_state=42,
    )
    model.fit(df_sub)
    pred = model.predict(df_sub)
    train_time = time.time() - start

    interval_df = pred.interval(0.9)
    n_pred = len(interval_df)
    y_true = df_sub[target].to_numpy()[-n_pred:]
    lower = interval_df["lower"].to_numpy()
    upper = interval_df["upper"].to_numpy()

    cov = coverage_score(y_true, lower, upper)
    wink = winkler_score(y_true, lower, upper, confidence=0.9)
    sharpness = (upper - lower).mean()

    return BenchmarkResult(
        dataset="",
        model="ConformalForecaster",
        n_samples=n_samples,
        train_time_sec=round(train_time, 3),
        coverage_90=round(cov, 4),
        sharpness_90=round(float(sharpness), 4),
        winkler_90=round(wink, 4),
    )


def run_benchmarks():
    """Run all benchmarks."""
    datasets = ["weather", "exchange_rate", "electricity"]
    results = []

    print("=" * 80)
    print("uncertainty_flow Trial Benchmark")
    print("=" * 80)

    for ds_name in datasets:
        print(f"\n[{ds_name}]")
        print("-" * 40)

        df, target = load_dataset(ds_name)
        print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"  Target: {target}")

        for n_samples in [500, 1000]:
            if len(df) < n_samples:
                continue

            print(f"\n  n_samples={n_samples}:")

            benchmarks = [
                ("QuantileForestForecaster", benchmark_quantile_forest),
                ("ConformalRegressor", benchmark_conformal_regressor),
                ("ConformalForecaster", benchmark_conformal_forecaster),
            ]

            for model_name, benchmark_fn in benchmarks:
                try:
                    result = benchmark_fn(df, target, n_samples)
                    result.dataset = ds_name
                    results.append(result)

                    print(f"    {result.model}:")
                    print(
                        f"      time={result.train_time_sec}s, coverage={result.coverage_90}, "
                        f"sharpness={result.sharpness_90}, winkler={result.winkler_90}"
                    )
                except Exception as e:
                    print(f"    {model_name}: ERROR - {e}")

    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)

    df_results = pl.DataFrame(
        [
            {
                "dataset": r.dataset,
                "model": r.model,
                "n_samples": r.n_samples,
                "train_time_sec": r.train_time_sec,
                "coverage_90": r.coverage_90,
                "sharpness_90": r.sharpness_90,
                "winkler_90": r.winkler_90,
            }
            for r in results
        ]
    )

    print(df_results)

    output_path = Path(__file__).parent.parent / "benchmark_results.csv"
    df_results.write_csv(output_path)
    print(f"\nResults saved to: {output_path}")

    return df_results


if __name__ == "__main__":
    run_benchmarks()
