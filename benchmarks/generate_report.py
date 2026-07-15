#!/usr/bin/env python3
"""Generate a comparison report from pipeline-native benchmark results.

Usage:
    uv run python benchmarks/generate_report.py
    uv run python benchmarks/generate_report.py --results-dir results --output results/report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from uncertainty_flow.benchmarking.registry import default_model_registry


def load_results(results_dir: str = "results") -> list[dict]:
    """Load typed ``PipelineRunResult`` JSON exports from a directory."""

    results: list[dict] = []
    results_path = Path(results_dir)

    for json_file in sorted(results_path.glob("*.json")):
        with json_file.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("model_results"), list):
            results.append(data)

    return results


def create_comparison_table(results: list[dict]) -> pl.DataFrame:
    """Flatten typed per-model results into the report table."""

    rows = []
    for data in results:
        manifest = data["manifest"]
        dataset_name = manifest["dataset_id"]
        dataset_domain = manifest["dataset_domain"]
        for result in data["model_results"]:
            metrics = result.get("metrics", {})
            rows.append(
                {
                    "dataset": dataset_name,
                    "domain": dataset_domain,
                    "model": result["model_id"],
                    "coverage_90": metrics.get("coverage_90", float("nan")),
                    "coverage_80": metrics.get("coverage_80", float("nan")),
                    "sharpness_90": metrics.get("sharpness_90", float("nan")),
                    "sharpness_80": metrics.get("sharpness_80", float("nan")),
                    "winkler_90": metrics.get("winkler_90", float("nan")),
                    "winkler_80": metrics.get("winkler_80", float("nan")),
                    "pinball": metrics.get("pinball", float("nan")),
                    "crps": metrics.get("crps", float("nan")),
                    "mae": metrics.get("mae", float("nan")),
                    "rmse": metrics.get("rmse", float("nan")),
                    "calibration_error_90": metrics.get("calibration_error_90", float("nan")),
                    "train_time_sec": result["train_time_sec"],
                    "timing_mean": result["train_time_sec"],
                    "timing_std": 0.0,
                    "memory_delta_mb": 0.0,
                    "n_runs": 1,
                }
            )
    return pl.DataFrame(rows)


def generate_report(df: pl.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("CONSOLIDATED BENCHMARK REPORT")
    lines.append("=" * 100)

    regression_models = {
        "linear-regression",
        "ridge-regression",
        "random-forest",
        "gradient-boosting",
    }
    baseline_models = {"naive-forecast", "moving-average"}
    uf_models = set(default_model_registry().names()) - regression_models - baseline_models

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
                f"{row.get('calibration_error_90', float('nan')):>7.4f} "
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
            pl.col("calibration_error_90").mean().alias("avg_cal_err"),
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
        print("No pipeline-native JSON results found. Run benchmarks first.")
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
