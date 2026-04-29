#!/usr/bin/env python3
"""Ingest sample datasets from Hugging Face and convert to Parquet."""

from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from datasets import load_dataset


def get_numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype != pl.String]


def _save(df: pl.DataFrame, output_path: Path) -> None:
    numeric_cols = get_numeric_cols(df)
    df = df.select(numeric_cols)
    df.write_parquet(output_path)
    print(f"  -> {output_path}: {len(df):,} rows, {len(df.columns)} cols")


# ---------------------------------------------------------------------------
# P0: Original time-series datasets
# ---------------------------------------------------------------------------


def ingest_weather(output_dir: Path) -> None:
    print("Downloading weather dataset...")
    ds = load_dataset("ts-arena/weather", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))
    _save(df, output_dir / "weather.parquet")


def ingest_exchange_rate(output_dir: Path) -> None:
    print("Downloading exchange_rate dataset...")
    ds = load_dataset("ts-arena/exchange_rate", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))
    _save(df, output_dir / "exchange_rate.parquet")


def ingest_electricity(output_dir: Path) -> None:
    print("Downloading electricity dataset...")
    ds = load_dataset("lalababa/Time-Series-Library", "electricity", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))
    _save(df, output_dir / "electricity.parquet")


# ---------------------------------------------------------------------------
# P0: UCI tabular datasets (heteroscedastic / non-Gaussian)
# ---------------------------------------------------------------------------


def ingest_concrete(output_dir: Path) -> None:
    print("Downloading concrete compressive strength dataset...")
    url = (
        "https://raw.githubusercontent.com/stedy/"
        "Machine-Learning-with-R-datasets/master/concrete.csv"
    )
    df = pl.read_csv(url, infer_schema_length=2000)
    _save(df, output_dir / "concrete.parquet")


def ingest_wine_quality(output_dir: Path) -> None:
    print("Downloading wine quality dataset...")
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    df = pl.read_csv(url, separator=";", infer_schema_length=10000)
    _save(df, output_dir / "wine_quality.parquet")


def ingest_energy_efficiency(output_dir: Path) -> None:
    print("Downloading energy efficiency dataset...")
    from sklearn.datasets import fetch_openml

    raw = fetch_openml(name="energy-efficiency", version=1, as_frame=True)
    frame = raw.frame.copy()
    frame["y1"] = frame["y1"].astype(float)
    frame["y2"] = frame["y2"].astype(float)
    df = pl.from_pandas(frame)
    _save(df, output_dir / "energy_efficiency.parquet")


# ---------------------------------------------------------------------------
# P1: Diverse time-series (regime changes, non-stationary, heavy tails)
# ---------------------------------------------------------------------------


def ingest_traffic(output_dir: Path, n_sensor_cols: int = 15) -> None:
    print(f"Downloading traffic dataset (keeping {n_sensor_cols} sensor cols)...")
    ds = load_dataset("ts-arena/traffic", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))
    numeric_cols = get_numeric_cols(df)
    priority = ["OT"] + [c for c in numeric_cols if c != "OT"]
    keep = priority[:n_sensor_cols]
    df = df.select(keep)
    _save(df, output_dir / "traffic.parquet")


def ingest_illness(output_dir: Path) -> None:
    print("Downloading illness dataset (regime-switching)...")
    ds = load_dataset("ts-arena/illness", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))
    _save(df, output_dir / "illness.parquet")


def ingest_beijingpm25(output_dir: Path) -> None:
    print("Downloading Beijing PM2.5 dataset (heavy tails + missing)...")
    try:
        ds = load_dataset("autogluon/chronos_datasets", "beijingpm25", split="train")
        df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))
        _save(df, output_dir / "beijingpm25_local.parquet")
    except Exception as exc:
        print(f"  SKIP (chronos beijingpm25 unavailable): {exc}")


# ---------------------------------------------------------------------------
# P2: Synthetic datasets with ground-truth uncertainty
# ---------------------------------------------------------------------------


def generate_synthetic_heteroscedastic(output_dir: Path, n: int = 5000, seed: int = 42) -> None:
    print(f"Generating synthetic heteroscedastic dataset (n={n:,})...")
    rng = np.random.default_rng(seed)
    n_features = 6
    x_mat = rng.standard_normal((n, n_features))
    coef = rng.standard_normal(n_features)
    noise_scale = 0.5 + 2.0 * np.abs(x_mat[:, 0])
    y = x_mat @ coef + noise_scale * rng.standard_normal(n)

    cols = [f"x{i}" for i in range(n_features)] + ["y"]
    arr = np.column_stack([x_mat, y])
    df = pl.DataFrame({c: arr[:, i] for i, c in enumerate(cols)})
    _save(df, output_dir / "synthetic_heteroscedastic.parquet")


def generate_synthetic_regime_switching(output_dir: Path, n: int = 5000, seed: int = 42) -> None:
    print(f"Generating synthetic regime-switching dataset (n={n:,})...")
    rng = np.random.default_rng(seed)
    regimes = np.zeros(n, dtype=int)
    regime = 0
    for i in range(1, n):
        if rng.random() < 0.01:
            regime = 1 - regime
        regimes[i] = regime

    means = np.where(regimes == 0, 0.0, 5.0)
    scales = np.where(regimes == 0, 1.0, 3.0)
    y = means + scales * rng.standard_normal(n)

    df = pl.DataFrame({"y": y, "regime": regimes})
    _save(df, output_dir / "synthetic_regime_switching.parquet")


def generate_synthetic_heavy_tailed(output_dir: Path, n: int = 5000, seed: int = 42) -> None:
    print(f"Generating synthetic heavy-tailed dataset (n={n:,})...")
    rng = np.random.default_rng(seed)
    n_features = 6
    df_t = 3.0
    x_mat = rng.standard_normal((n, n_features))
    coef = rng.standard_normal(n_features)
    noise = rng.standard_t(df_t, size=n)
    y = x_mat @ coef + noise

    cols = [f"x{i}" for i in range(n_features)] + ["y"]
    arr = np.column_stack([x_mat, y])
    df = pl.DataFrame({c: arr[:, i] for i, c in enumerate(cols)})
    _save(df, output_dir / "synthetic_heavy_tailed.parquet")


def generate_synthetic_multivariate(output_dir: Path, n: int = 5000, seed: int = 42) -> None:
    print(f"Generating synthetic multivariate target dataset (n={n:,})...")
    from scipy import stats as sp_stats

    rng = np.random.default_rng(seed)
    n_features = 4
    x_mat = rng.standard_normal((n, n_features))

    theta_clayton = 2.0
    u = rng.uniform(size=(n, 3))
    v1 = u[:, 0]
    w = u[:, 1]
    v2 = (
        v1 ** (-(theta_clayton + 1)) * (w ** (-(theta_clayton + 1) / theta_clayton))
        - v1 ** (-(theta_clayton + 1))
        + 1
    ) ** (-1 / (theta_clayton + 1))
    v3 = rng.uniform(size=n)

    y1 = x_mat[:, 0] + sp_stats.norm.ppf(v1) + rng.standard_normal(n) * 0.3
    y2 = x_mat[:, 1] + sp_stats.norm.ppf(v2) + rng.standard_normal(n) * 0.3
    y3 = sp_stats.norm.ppf(v3) + rng.standard_normal(n) * 0.5

    cols = [f"x{i}" for i in range(n_features)] + ["y1", "y2", "y3"]
    arr = np.column_stack([x_mat, y1, y2, y3])
    df = pl.DataFrame({c: arr[:, i] for i, c in enumerate(cols)})
    _save(df, output_dir / "synthetic_multivariate.parquet")


# ---------------------------------------------------------------------------
# P2: Derived financial volatility dataset
# ---------------------------------------------------------------------------


def generate_financial_volatility(output_dir: Path) -> None:
    print("Generating financial volatility dataset from exchange_rate data...")
    exchange_path = output_dir / "exchange_rate.parquet"
    if not exchange_path.exists():
        print("  SKIP: exchange_rate.parquet not found (run --group original first)")
        return

    df_raw = pl.read_parquet(exchange_path)
    rate_cols = [c for c in df_raw.columns if c != "timestamp_idx"]
    df = df_raw.select(rate_cols).cast(pl.Float64)

    for c in rate_cols:
        vals = df[c].to_numpy()
        diffs = np.zeros_like(vals)
        diffs[1:] = np.diff(vals)
        df = df.with_columns(pl.Series(c + "_diff", diffs))

    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    for c in diff_cols:
        df = df.with_columns(pl.col(c).rolling_std(window_size=20).alias(c + "_vol20"))
        df = df.with_columns(pl.col(c).rolling_std(window_size=5).alias(c + "_vol5"))

    feature_cols = [c for c in df.columns if c != "OT_diff" and not c.startswith("OT_diff_")]
    df_final = df.select(feature_cols + ["OT_diff"])
    df_final = df_final.slice(20)
    df_final = df_final.fill_null(strategy="forward").fill_null(strategy="backward")
    _save(df_final, output_dir / "financial_volatility.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ingest_fraud(output_dir: Path, n_samples: int = 50_000) -> None:
    print(f"Downloading fraud detection dataset (sampling {n_samples:,} rows)...")
    try:
        ds = load_dataset(
            "vitaliy-sharandin/synthetic-fraud-detection",
            split="train",
            streaming=True,
        )
        rows = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            rows.append(row)
        if not rows:
            print("  SKIP: no rows received from fraud dataset")
            return
        df = pl.DataFrame(rows)
        df = df.with_columns(pl.col("type").cast(pl.Categorical).to_physical().alias("type_code"))
        numeric_cols = get_numeric_cols(df)
        df = df.select(numeric_cols)
        _save(df, output_dir / "fraud.parquet")
    except Exception as exc:
        print(f"  SKIP (fraud dataset unavailable): {exc}")


def ingest_climsim(output_dir: Path, n_samples: int = 10_000) -> None:
    print(f"Downloading ClimSim dataset (sampling {n_samples:,} rows via streaming)...")
    try:
        ds = load_dataset("LEAP/ClimSim_low-res", split="train", streaming=True)
        rows = []
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            rows.append(row)
        if not rows:
            print("  SKIP: no rows received from ClimSim")
            return
        df = pl.DataFrame(rows)
        numeric_cols = get_numeric_cols(df)
        df = df.select(numeric_cols)
        _save(df, output_dir / "climsim.parquet")
    except Exception as exc:
        print(f"  SKIP (ClimSim unavailable or too slow): {exc}")


def convert_csv_to_parquet(csv_path: Path, output_dir: Path) -> None:
    print(f"Converting {csv_path} to Parquet...")
    df = pl.read_csv(csv_path)
    numeric_cols = get_numeric_cols(df)
    df = df.select(numeric_cols)
    output_path = output_dir / f"{csv_path.stem}.parquet"
    df.write_parquet(output_path)
    print(f"  -> {output_path}: {len(df):,} rows, {len(df.columns)} cols")


ORIGINAL_INGESTERS = [
    ingest_weather,
    ingest_exchange_rate,
    ingest_electricity,
]

TABULAR_INGESTERS = [
    ingest_concrete,
    ingest_wine_quality,
    ingest_energy_efficiency,
]

TS_DIVERSE_INGESTERS = [
    ingest_traffic,
    ingest_illness,
    ingest_beijingpm25,
    ingest_climsim,
    ingest_fraud,
]

SYNTHETIC_GENERATORS = [
    generate_synthetic_heteroscedastic,
    generate_synthetic_regime_switching,
    generate_synthetic_heavy_tailed,
    generate_synthetic_multivariate,
    generate_financial_volatility,
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest benchmark datasets")
    parser.add_argument(
        "--group",
        choices=["all", "original", "tabular", "ts-diverse", "synthetic"],
        default="all",
        help="Which dataset group to ingest (default: all)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)

    ingesters: list = []
    if args.group in ("all", "original"):
        ingesters.extend(ORIGINAL_INGESTERS)
    if args.group in ("all", "tabular"):
        ingesters.extend(TABULAR_INGESTERS)
    if args.group in ("all", "ts-diverse"):
        ingesters.extend(TS_DIVERSE_INGESTERS)
    if args.group in ("all", "synthetic"):
        ingesters.extend(SYNTHETIC_GENERATORS)

    print("=" * 60)
    print(f"Ingesting datasets (group={args.group})")
    print("=" * 60)

    for ingest_fn in ingesters:
        try:
            ingest_fn(data_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print("\n" + "=" * 60)
        print("Converting existing CSV files to Parquet")
        print("=" * 60)
        for csv_path in csv_files:
            convert_csv_to_parquet(csv_path, data_dir)

        print("\n" + "=" * 60)
        print("Removing CSV files")
        print("=" * 60)
        for csv_path in csv_files:
            csv_path.unlink()
            print(f"  Removed: {csv_path.name}")

    print("\n" + "=" * 60)
    print("Dataset ingestion complete!")
    print("=" * 60)

    for f in sorted(data_dir.glob("*.parquet")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
