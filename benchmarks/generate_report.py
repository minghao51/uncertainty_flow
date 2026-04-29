#!/usr/bin/env python3
"""Generate a comparison report from consolidated benchmark results.

Usage:
    uv run python benchmarks/generate_report.py
    uv run python benchmarks/generate_report.py --results-dir results --output results/report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from run_benchmarks import UF_MODELS


def load_results(results_dir: str = "results") -> dict:
    results = {}
    results_path = Path(results_dir)

    for json_file in sorted(results_path.glob("consolidated_*.json")):
        if "_all.json" in str(json_file):
            continue
        with open(json_file) as f:
            data = json.load(f)
            dataset_name = data["metadata"]["dataset"]
            results[dataset_name] = data

    return results


def create_comparison_table(results: dict) -> pl.DataFrame:
    rows = []
    for dataset_name, data in results.items():
        metadata = data["metadata"]
        for r in data["results"]:
            rows.append(
                {
                    "dataset": dataset_name,
                    "domain": metadata["domain"],
                    "model": r["model"],
                    "coverage_90": r["coverage_90"],
                    "coverage_80": r.get("coverage_80", 0),
                    "sharpness_90": r["sharpness_90"],
                    "sharpness_80": r.get("sharpness_80", 0),
                    "winkler_90": r["winkler_90"],
                    "winkler_80": r.get("winkler_80", 0),
                    "pinball_loss": r.get("pinball_loss", 0),
                    "crps": r.get("crps", 0),
                    "mae": r.get("mae", 0),
                    "rmse": r.get("rmse", 0),
                    "calibration_error": r.get("calibration_error", 0),
                    "train_time_sec": r["train_time_sec"],
                    "timing_mean": r.get("timing_mean", r["train_time_sec"]),
                    "timing_std": r.get("std", 0),
                    "memory_delta_mb": r.get("memory_delta_mb", 0),
                    "n_runs": r.get("n_runs", 1),
                }
            )
    return pl.DataFrame(rows)


def generate_report(df: pl.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("CONSOLIDATED BENCHMARK REPORT")
    lines.append("=" * 100)

    uf_models = set(UF_MODELS)
    regression_models = {
        "linear-regression",
        "ridge-regression",
        "random-forest",
        "gradient-boosting",
    }
    baseline_models = {"naive-forecast", "moving-average"}

    for dataset in df["dataset"].unique().to_list():
        ds_df = df.filter(pl.col("dataset") == dataset)
        domain = ds_df["domain"][0]
        lines.append(f"\n{'─' * 100}")
        lines.append(f"  {dataset} ({domain})")
        lines.append(f"{'─' * 100}")

        best_winkler = ds_df.sort("winkler_90").row(0, named=True)
        ds_df_cov = ds_df.with_columns((pl.col("coverage_90") - 0.90).abs().alias("cov_dev"))
        best_cov = ds_df_cov.sort("cov_dev").row(0, named=True)
        fastest = ds_df.sort("timing_mean").row(0, named=True)
        best_mae_row = ds_df.sort("mae").row(0, named=True)

        lines.append(
            f"  Best Winkler@90%: {best_winkler['model']} ({best_winkler['winkler_90']:.4f})"
        )
        lines.append(f"  Best Coverage@90%: {best_cov['model']} ({best_cov['coverage_90']:.4f})")
        lines.append(f"  Best MAE: {best_mae_row['model']} ({best_mae_row['mae']:.4f})")
        lines.append(
            f"  Fastest: {fastest['model']} "
            f"({fastest['timing_mean']:.4f}s ± {fastest['timing_std']:.4f})"
        )

        lines.append(
            f"\n  {'Model':<24} {'Cov@90':>7} "
            f"{'Wink@90':>9} {'CRPS':>9} {'MAE':>9} "
            f"{'RMSE':>9} {'CalErr':>7} {'Time':>10} {'±std':>8} {'MemMB':>7}"
        )
        lines.append(f"  {'─' * 100}")
        for row in ds_df.sort("winkler_90").iter_rows(named=True):
            lines.append(
                f"  {row['model']:<24} {row['coverage_90']:>7.4f} "
                f"{row['winkler_90']:>9.4f} {row.get('crps', 0):>9.4f} "
                f"{row.get('mae', 0):>9.4f} {row.get('rmse', 0):>9.4f} "
                f"{row.get('calibration_error', 0):>7.4f} "
                f"{row['timing_mean']:>10.4f} "
                f"{row['timing_std']:>8.4f} {row.get('memory_delta_mb', 0):>7.1f}"
            )

    lines.append(f"\n{'=' * 100}")
    lines.append("OVERALL RANKINGS (by mean Winkler@90%)")
    lines.append(f"{'=' * 100}")

    avg_rank = (
        df.group_by("model")
        .agg(
            pl.col("winkler_90").mean().alias("avg_winkler_90"),
            pl.col("mae").mean().alias("avg_mae"),
            pl.col("crps").mean().alias("avg_crps"),
            pl.col("calibration_error").mean().alias("avg_cal_err"),
        )
        .sort("avg_winkler_90")
    )
    for i, row in enumerate(avg_rank.iter_rows(named=True), 1):
        lines.append(
            f"  {i}. {row['model']:<24} "
            f"wink={row['avg_winkler_90']:.4f} mae={row['avg_mae']:.4f} "
            f"crps={row['avg_crps']:.4f} cal_err={row['avg_cal_err']:.4f}"
        )

    lines.append(f"\n{'=' * 100}")
    lines.append("CATEGORY COMPARISON")
    lines.append(f"{'=' * 100}")
    for cat_name, models in [
        ("Uncertainty Flow", uf_models),
        ("Regression Baselines", regression_models),
        ("Simple Baselines", baseline_models),
    ]:
        lines.append(f"\n  {cat_name}:")
        cat_df = df.filter(pl.col("model").is_in(models))
        for row in (
            cat_df.group_by("model")
            .agg(
                pl.col("coverage_90").mean().alias("avg_cov"),
                pl.col("winkler_90").mean().alias("avg_wink"),
                pl.col("mae").mean().alias("avg_mae"),
                pl.col("crps").mean().alias("avg_crps"),
                pl.col("timing_mean").mean().alias("avg_time"),
            )
            .sort("avg_wink")
            .iter_rows(named=True)
        ):
            lines.append(
                f"    {row['model']:<24} cov={row['avg_cov']:.3f} "
                f"wink={row['avg_wink']:.4f} mae={row['avg_mae']:.4f} "
                f"crps={row['avg_crps']:.4f} time={row['avg_time']:.4f}s"
            )

    lines.append("\n" + "=" * 100)
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print("No consolidated results found. Run benchmarks first.")
        return

    df = create_comparison_table(results)

    csv_path = Path(args.results_dir) / "consolidated_comparison.csv"
    df.write_csv(csv_path)
    print(f"Comparison table saved to: {csv_path}")

    report = generate_report(df)
    if args.output:
        Path(args.output).write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
