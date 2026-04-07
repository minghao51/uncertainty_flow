#!/usr/bin/env python3
"""Generate comparison analysis report from benchmark results.

Run with: uv run python scripts/generate_report.py
"""

import json
from pathlib import Path

import polars as pl


def load_results(results_dir: str = "results") -> dict:
    """Load all benchmark results from JSON files."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("comprehensive_v2_*.json"):
        if "_all.json" in str(json_file):
            continue

        with open(json_file) as f:
            data = json.load(f)
            dataset_name = data["metadata"]["dataset"]
            results[dataset_name] = data

    return results


def create_comparison_table(results: dict) -> pl.DataFrame:
    """Create a comparison table from benchmark results."""
    rows = []

    for dataset_name, data in results.items():
        metadata = data["metadata"]
        for model_result in data["results"]:
            rows.append(
                {
                    "dataset": dataset_name,
                    "domain": metadata["domain"],
                    "model": model_result["model"],
                    "coverage_90": model_result["coverage_90"],
                    "coverage_80": model_result["coverage_80"],
                    "sharpness_90": model_result["sharpness_90"],
                    "sharpness_80": model_result["sharpness_80"],
                    "winkler_90": model_result["winkler_90"],
                    "winkler_80": model_result["winkler_80"],
                    "pinball_loss": model_result["pinball_loss"],
                    "train_time_sec": model_result["train_time_sec"],
                }
            )

    return pl.DataFrame(rows)


def rank_models(df: pl.DataFrame, metric: str = "winkler_90") -> pl.DataFrame:
    """Rank models by a specific metric."""
    return (
        df.sort("dataset", metric)
        .with_columns(pl.col(metric).rank(method="dense").over("dataset").alias(f"{metric}_rank"))
        .select(
            "dataset",
            "domain",
            "model",
            metric,
            f"{metric}_rank",
        )
        .sort("dataset", f"{metric}_rank")
    )


def generate_summary_report(df: pl.DataFrame) -> str:
    """Generate a summary report."""
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE BENCHMARK COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")

    # Group by model category
    uf_models = [
        "quantile-forest",
        "conformal-regressor",
        "conformal-forecaster",
    ]
    regression_models = [
        "linear-regression",
        "ridge-regression",
        "random-forest",
        "gradient-boosting",
    ]
    baseline_models = [
        "naive-forecast",
        "moving-average",
    ]

    # Summary by dataset
    report.append("RESULTS BY DATASET")
    report.append("-" * 80)

    for dataset in df["dataset"].unique():
        ds_df = df.filter(pl.col("dataset") == dataset)
        domain = ds_df["domain"][0]

        report.append(f"\n{dataset} ({domain})")
        report.append("~" * 40)

        # Best model by Winkler score (lower is better)
        best_winkler = ds_df.sort("winkler_90").head(1)
        report.append(
            f"\n  Best Winkler @ 90%: {best_winkler['model'][0]} "
            f"({best_winkler['winkler_90'][0]:.4f})"
        )

        # Best coverage (closest to 0.90)
        ds_df = ds_df.with_columns(
            (pl.col("coverage_90") - 0.90).abs().alias("coverage_90_deviation")
        )
        best_coverage = ds_df.sort("coverage_90_deviation").head(1)
        report.append(
            f"  Best Coverage @ 90%: {best_coverage['model'][0]} "
            f"({best_coverage['coverage_90'][0]:.4f})"
        )

        # Fastest model
        fastest = ds_df.sort("train_time_sec").head(1)
        report.append(
            f"  Fastest Training: {fastest['model'][0]} ({fastest['train_time_sec'][0]:.3f}s)"
        )

    report.append("")
    report.append("")

    # Overall rankings
    report.append("OVERALL MODEL RANKINGS (by average Winkler @ 90%)")
    report.append("-" * 80)

    avg_rankings = (
        df.group_by("model")
        .agg(pl.col("winkler_90").mean().alias("avg_winkler_90"))
        .sort("avg_winkler_90")
        .with_columns(pl.col("avg_winkler_90").rank(method="dense").alias("rank"))
    )

    report.append("")
    for row in avg_rankings.iter_rows():
        model, avg_winkler, rank = row
        report.append(f"  {rank:.0f}. {model}: {avg_winkler:.4f}")

    report.append("")
    report.append("")

    # Category comparison
    report.append("COMPARISON BY MODEL CATEGORY")
    report.append("-" * 80)

    for category_name, models in [
        ("Uncertainty Flow Models", uf_models),
        ("Regression Baselines", regression_models),
        ("Simple Baselines", baseline_models),
    ]:
        report.append(f"\n{category_name}:")
        cat_df = df.filter(pl.col("model").is_in(models))
        avg_metrics = (
            cat_df.group_by("model")
            .agg(
                pl.col("coverage_90").mean().alias("avg_coverage_90"),
                pl.col("sharpness_90").mean().alias("avg_sharpness_90"),
                pl.col("winkler_90").mean().alias("avg_winkler_90"),
                pl.col("train_time_sec").mean().alias("avg_train_time"),
            )
            .sort("avg_winkler_90")
        )

        for row in avg_metrics.iter_rows():
            model, cov, sharp, wink, time = row
            report.append(
                f"    {model}: cov={cov:.3f}, sharp={sharp:.4f}, wink={wink:.4f}, time={time:.3f}s"
            )

    report.append("")
    report.append("")

    # Key findings
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    report.append("")

    # Find best uncertainty flow model
    uf_df = df.filter(pl.col("model").is_in(uf_models))
    best_uf = (
        uf_df.group_by("model")
        .agg(pl.col("winkler_90").mean().alias("avg_winkler_90"))
        .sort("avg_winkler_90")
        .head(1)
    )

    # Find best baseline
    baseline_df = df.filter(pl.col("model").is_in(regression_models + baseline_models))
    best_baseline = (
        baseline_df.group_by("model")
        .agg(pl.col("winkler_90").mean().alias("avg_winkler_90"))
        .sort("avg_winkler_90")
        .head(1)
    )

    report.append(
        f"1. Best Uncertainty Flow Model: {best_uf['model'][0]} "
        f"(avg Winkler @ 90%: {best_uf['avg_winkler_90'][0]:.4f})"
    )
    report.append(
        f"2. Best Baseline Model: {best_baseline['model'][0]} "
        f"(avg Winkler @ 90%: {best_baseline['avg_winkler_90'][0]:.4f})"
    )

    # Coverage analysis
    report.append("")
    report.append("COVERAGE ANALYSIS")
    report.append("-" * 80)

    coverage_summary = (
        df.group_by("model")
        .agg(
            pl.col("coverage_90").mean().alias("avg_coverage_90"),
            pl.col("coverage_80").mean().alias("avg_coverage_80"),
        )
        .sort("avg_coverage_90", descending=True)
    )

    report.append("")
    report.append("  Models ranked by average coverage @ 90%:")
    for row in coverage_summary.iter_rows():
        model, cov90, cov80 = row
        target_dev_90 = abs(cov90 - 0.90)
        target_dev_80 = abs(cov80 - 0.80)
        report.append(
            f"    {model}: cov90={cov90:.3f} (dev={target_dev_90:.3f}), "
            f"cov80={cov80:.3f} (dev={target_dev_80:.3f})"
        )

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for the report",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} datasets")

    # Create comparison table
    df = create_comparison_table(results)

    # Save CSV
    csv_path = Path(args.results_dir) / "comparison_table.csv"
    df.write_csv(csv_path)
    print(f"Comparison table saved to: {csv_path}")

    # Generate report
    report = generate_summary_report(df)

    # Output report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
