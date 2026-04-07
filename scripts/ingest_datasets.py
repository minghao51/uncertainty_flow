#!/usr/bin/env python3
"""Ingest sample datasets from Hugging Face and convert to Parquet."""

from pathlib import Path
from typing import cast

import polars as pl
from datasets import load_dataset


def get_numeric_cols(df: pl.DataFrame) -> list[str]:
    """Get only numeric columns for sklearn compatibility."""
    return [c for c in df.columns if df[c].dtype != pl.String]


def ingest_weather(output_dir: Path) -> None:
    """Download and convert weather dataset."""
    print("Downloading weather dataset...")
    ds = load_dataset("ts-arena/weather", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))

    numeric_cols = get_numeric_cols(df)
    df = df.select(numeric_cols)

    output_path = output_dir / "weather.parquet"
    df.write_parquet(output_path)
    print(f"  -> {output_path}: {len(df):,} rows, {len(df.columns)} cols")


def ingest_exchange_rate(output_dir: Path) -> None:
    """Download and convert exchange rate dataset."""
    print("Downloading exchange_rate dataset...")
    ds = load_dataset("ts-arena/exchange_rate", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))

    numeric_cols = get_numeric_cols(df)
    df = df.select(numeric_cols)

    output_path = output_dir / "exchange_rate.parquet"
    df.write_parquet(output_path)
    print(f"  -> {output_path}: {len(df):,} rows, {len(df.columns)} cols")


def ingest_electricity(output_dir: Path) -> None:
    """Download and convert electricity dataset."""
    print("Downloading electricity dataset...")
    ds = load_dataset("lalababa/Time-Series-Library", "electricity", split="train")
    df = cast(pl.DataFrame, pl.from_arrow(ds.data.table))

    numeric_cols = get_numeric_cols(df)
    df = df.select(numeric_cols)

    output_path = output_dir / "electricity.parquet"
    df.write_parquet(output_path)
    print(f"  -> {output_path}: {len(df):,} rows, {len(df.columns)} cols")


def convert_csv_to_parquet(csv_path: Path, output_dir: Path) -> None:
    """Convert existing CSV to Parquet."""
    print(f"Converting {csv_path} to Parquet...")
    df = pl.read_csv(csv_path)

    numeric_cols = get_numeric_cols(df)
    df = df.select(numeric_cols)

    output_path = output_dir / f"{csv_path.stem}.parquet"
    df.write_parquet(output_path)
    print(f"  -> {output_path}: {len(df):,} rows, {len(df.columns)} cols")


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Ingesting sample datasets from Hugging Face")
    print("=" * 60)

    ingest_weather(data_dir)
    ingest_exchange_rate(data_dir)
    ingest_electricity(data_dir)

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
