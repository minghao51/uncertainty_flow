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
                    "train_time_sec": r["train_time_sec"],
                    "timing_mean": r.get("mean", r["train_time_sec"]),
                    "timing_std": r.get("std", 0),
                    "n_runs": r.get("n_runs", 1),
                }
            )
    return pl.DataFrame(rows)


def generate_report(df: pl.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("CONSOLIDATED BENCHMARK REPORT")
    lines.append("=" * 90)

    uf_models = {"quantile-forest", "conformal-regressor", "conformal-forecaster"}
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
        lines.append(f"
{'─' * 90}")
        lines.append(f"  {dataset} ({domain})")
        lines.append(f"{'─' * 90}")

        best_winkler = ds_df.sort("winkler_90").row(0, named=True)
        ds_df_cov = ds_df.with_columns((pl.col("coverage_90") - 0.90).abs().alias("cov_dev"))
        best_cov = ds_df_cov.sort("cov_dev").row(0, named=True)
        fastest = ds_df.sort("timing_mean").row(0, named=True)

        lines.append(
            f"  Best Winkler@90%: {best_winkler['model']} ({best_winkler['winkler_90']:.4f})"
        )
        lines.append(f"  Best Coverage@90%: {best_cov['model']} ({best_cov['coverage_90']:.4f})")
        lines.append(
            f"  Fastest: {fastest['model']} "
            f"({fastest['timing_mean']:.4f}s ± {fastest['timing_std']:.4f})"
        )

        lines.append(
            f"
  {'Model':<22} {'Cov@90':>7} "
            f"{'Wink@90':>9} {'Sharp@90':>9} "
            f"{'Time':>10} {'±std':>8} {'Runs':>5}"
        )
        lines.append(f"  {'─' * 70}")
        for row in ds_df.sort("winkler_90").iter_rows(named=True):
            lines.append(
                f"  {row['model']:<22} {row['coverage_90']:>7.4f} {row['winkler_90']:>9.4f} "
                f"{row['sharpness_90']:>9.4f} {row['timing_mean']:>10.4f} "
                f"{row['timing_std']:>8.4f} {row['n_runs']:>5}"
            )

    lines.append(f"
{'=' * 90}")
    lines.append("OVERALL RANKINGS (by mean Winkler@90%)")
    lines.append(f"{'=' * 90}")

    avg_rank = (
        df.group_by("model")
        .agg(pl.col("winkler_90").mean().alias("avg_winkler_90"))
        .sort("avg_winkler_90")
    )
    for i, row in enumerate(avg_rank.iter_rows(named=True), 1):
        lines.append(f"  {i}. {row['model']:<22} {row['avg_winkler_90']:.4f}")

    lines.append(f"
{'=' * 90}")
    lines.append("CATEGORY COMPARISON")
    lines.append(f"{'=' * 90}")
    for cat_name, models in [
        ("Uncertainty Flow", uf_models),
        ("Regression Baselines", regression_models),
        ("Simple Baselines", baseline_models),
    ]:
        lines.append(f"
  {cat_name}:")
        cat_df = df.filter(pl.col("model").is_in(models))
        for row in (
            cat_df.group_by("model")
            .agg(
                pl.col("coverage_90").mean().alias("avg_cov"),
                pl.col("winkler_90").mean().alias("avg_wink"),
                pl.col("timing_mean").mean().alias("avg_time"),
            )
            .sort("avg_wink")
            .iter_rows(named=True)
        ):
            lines.append(
                f"    {row['model']:<22} cov={row['avg_cov']:.3f} "
                f"wink={row['avg_wink']:.4f} time={row['avg_time']:.4f}s"
            )

    lines.append("
" + "=" * 90)
    return "
".join(lines)


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
