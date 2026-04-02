#!/usr/bin/env python3
"""CLI for uncertainty_flow benchmarking."""

import sys
from pathlib import Path
from typing import Any

import click

from uncertainty_flow import __version__
from uncertainty_flow.benchmarking import (
    AVAILABLE_DATASETS,
    BenchmarkConfig,
    BenchmarkRunner,
    TuningResult,
    auto_tune,
)
from uncertainty_flow.benchmarking.datasets import download_dataset


@click.group()
@click.version_option(version=__version__, prog_name="uncertainty-flow")
def cli() -> None:
    """uncertainty-flow: Probabilistic forecasting and uncertainty quantification."""
    pass


@cli.command()
@click.option(
    "--domain",
    type=str,
    default=None,
    help="Filter by domain (e.g., 'Energy', 'Finance', 'Climate')",
)
def list_datasets_cmd(domain: str | None) -> None:
    """List available datasets for benchmarking."""
    if domain:
        datasets = [ds for ds in AVAILABLE_DATASETS.values() if ds.domain == domain]
        if not datasets:
            click.echo(f"No datasets found for domain '{domain}'.", err=True)
            available = sorted(set(ds.domain for ds in AVAILABLE_DATASETS.values()))
            click.echo(f"Available domains: {', '.join(available)}")
            sys.exit(1)
    else:
        datasets = list(AVAILABLE_DATASETS.values())

    click.echo(f"\nAvailable datasets ({len(datasets)} total):\n")
    click.echo(f"{'Name':<35} {'Domain':<15} {'HF Path'}")
    click.echo("-" * 100)

    for ds in sorted(datasets, key=lambda x: x.name):
        hf_path = f"{ds.hf_path}/{ds.subset}" if ds.subset else ds.hf_path
        click.echo(f"{ds.name:<35} {ds.domain:<15} {hf_path}")

    click.echo(f"\nTotal: {len(datasets)} datasets")


@cli.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help=(
        "Dataset name or HuggingFace path "
        "(e.g., 'm4_daily' or 'autogluon/chronos_datasets/m4_daily')"
    ),
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="all",
    help=(
        "Comma-separated model names or 'all'. "
        "Options: quantile-forest, conformal-regressor, conformal-forecaster"
    ),
)
@click.option(
    "--n-samples",
    "-n",
    type=int,
    default=1000,
    help="Number of samples to use from dataset (default: 1000)",
)
@click.option(
    "--horizon",
    "-h",
    type=int,
    default=3,
    help="Forecast horizon for time series models (default: 3)",
)
@click.option(
    "--n-estimators",
    "-e",
    type=int,
    default=30,
    help="Number of estimators for base models (default: 30)",
)
@click.option(
    "--target",
    "-t",
    type=str,
    default=None,
    help="Target column name (default: uses dataset default)",
)
@click.option(
    "--auto-tune/--no-auto-tune",
    default=True,
    help="Enable or disable auto-tuning (default: enabled)",
)
@click.option(
    "--target-coverage",
    "-c",
    type=float,
    default=0.9,
    help="Target coverage level for tuning (default: 0.9)",
)
@click.option(
    "--tune-samples",
    type=int,
    default=500,
    help="Number of samples to use for tuning (default: 500)",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help=(
        "Output file path (JSON and CSV will be created with this prefix, or .json/.csv extensions)"
    ),
)
@click.option(
    "--json-only",
    is_flag=True,
    help="Only output JSON, skip CSV",
)
@click.option(
    "--csv-only",
    is_flag=True,
    help="Only output CSV, skip JSON",
)
def benchmark(
    dataset: str,
    model: str,
    n_samples: int,
    horizon: int,
    n_estimators: int,
    target: str | None,
    auto_tune: bool,
    target_coverage: float,
    tune_samples: int,
    output: str | None,
    json_only: bool,
    csv_only: bool,
) -> None:
    """Run benchmark on a dataset with optional auto-tuning.

    Auto-tuning is enabled by default and will find optimal hyperparameters
    for each model to achieve the target coverage level.

    Examples:

        # List available datasets
        uncertainty-flow list-datasets

        # Run all models with auto-tuning (default)
        uncertainty-flow benchmark --dataset weather

        # Run without auto-tuning
        uncertainty-flow benchmark --dataset weather --no-auto-tune

        # Run specific models
        uncertainty-flow benchmark --dataset m4_daily \\
            --model quantile-forest,conformal-regressor

        # Run with custom settings
        uncertainty-flow benchmark --dataset electricity \\
            --n-samples 5000 --horizon 6 --n-estimators 50
    """
    config = BenchmarkConfig(
        dataset_name=dataset,
        n_samples=n_samples,
        horizon=horizon,
        n_estimators=n_estimators,
        target_column=target,
        auto_tune=auto_tune,
        target_coverage=target_coverage,
        tune_samples=tune_samples,
    )

    if model == "all":
        model_names = None
    else:
        model_names = [m.strip() for m in model.split(",")]
        valid_models = {"quantile-forest", "conformal-regressor", "conformal-forecaster"}
        for m in model_names:
            if m not in valid_models:
                click.echo(
                    f"Error: Unknown model '{m}'. Valid options: {', '.join(valid_models)}",
                    err=True,
                )
                sys.exit(1)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Benchmark: {dataset}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Samples: {n_samples}")
    click.echo(f"  Horizon: {horizon}")
    click.echo(f"  Estimators: {n_estimators}")
    if target:
        click.echo(f"  Target: {target}")
    click.echo(f"  Models: {model if model != 'all' else 'all'}")
    click.echo()

    runner = BenchmarkRunner(config)

    try:
        click.echo("Loading dataset...")
        runner.load_data()
        assert runner.df is not None, "Failed to load dataset"
        assert runner.ds_info is not None, "Failed to get dataset info"
        click.echo(f"  Loaded: {len(runner.df):,} rows, {len(runner.df.columns)} columns")
        click.echo(f"  Domain: {runner.ds_info.domain}")
        click.echo(f"  Target: {runner.target}")
        click.echo()

        result = runner.run_all(model_names)

        click.echo(f"\n{'=' * 60}")
        click.echo("Results")
        click.echo(f"{'=' * 60}\n")

        for r in result.models:
            click.echo(f"  {r.model_name}:")
            click.echo(f"    Coverage @ 90%: {r.coverage_90:.4f}")
            click.echo(f"    Coverage @ 80%: {r.coverage_80:.4f}")
            click.echo(f"    Sharpness @ 90%: {r.sharpness_90:.4f}")
            click.echo(f"    Sharpness @ 80%: {r.sharpness_80:.4f}")
            click.echo(f"    Winkler @ 90%: {r.winkler_90:.4f}")
            click.echo(f"    Winkler @ 80%: {r.winkler_80:.4f}")
            click.echo(f"    Train time: {r.train_time_sec}s")
            click.echo()

        if output:
            output_path = Path(output)
            json_path = output_path.with_suffix(".json") if output_path.suffix else output_path
            csv_path = (
                output_path.with_suffix(".csv")
                if output_path.suffix
                else output_path.with_name(f"{output_path.name}.csv")
            )

            if not json_only:
                runner.save_json(json_path)
                click.echo(f"JSON results saved to: {json_path}")

            if not csv_only:
                runner.save_csv(csv_path)
                click.echo(f"CSV results saved to: {csv_path}")
        else:
            default_json = Path("benchmark_results.json")
            default_csv = Path("benchmark_results.csv")
            if not json_only:
                runner.save_json(default_json)
                click.echo(f"JSON results saved to: {default_json}")
            if not csv_only:
                runner.save_csv(default_csv)
                click.echo(f"CSV results saved to: {default_csv}")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="Dataset name or HuggingFace path",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="all",
    help=(
        "Comma-separated model names or 'all'. "
        "Options: quantile-forest, conformal-regressor, conformal-forecaster"
    ),
)
@click.option(
    "--n-samples",
    "-n",
    type=int,
    default=1000,
    help="Number of samples to use for tuning (default: 1000)",
)
@click.option(
    "--target-coverage",
    "-c",
    type=float,
    default=0.9,
    help="Target coverage level (default: 0.9)",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Output file for tuned parameters",
)
def tune(
    dataset: str,
    model: str,
    n_samples: int,
    target_coverage: float,
    output: str | None,
) -> None:
    """Automatically tune hyperparameters for optimal coverage.

    Examples:

        # Tune all models on weather dataset
        uncertainty-flow tune --dataset weather

        # Tune specific model
        uncertainty-flow tune --dataset weather --model conformal-regressor

        # Tune with custom target coverage
        uncertainty-flow tune --dataset weather --target-coverage 0.8
    """
    import json

    if model == "all":
        model_names = ["quantile-forest", "conformal-regressor", "conformal-forecaster"]
    else:
        model_names = [m.strip() for m in model.split(",")]
        valid_models = {"quantile-forest", "conformal-regressor", "conformal-forecaster"}
        for m in model_names:
            if m not in valid_models:
                click.echo(
                    f"Error: Unknown model '{m}'. Valid options: {', '.join(valid_models)}",
                    err=True,
                )
                sys.exit(1)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Auto-tune: {dataset}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Samples: {n_samples}")
    click.echo(f"  Target coverage: {target_coverage}")
    click.echo(f"  Models: {', '.join(model_names)}")
    click.echo()

    results: list[TuningResult] = []
    best_configs: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        try:
            click.echo(f"[{model_name}]")
            result = auto_tune(
                dataset_name=dataset,
                model_name=model_name,
                n_samples=n_samples,
                target_coverage=target_coverage,
            )
            results.append(result)
            best_configs[model_name] = result.best_params

            click.echo(f"  Best params: {result.best_params}")
            click.echo(f"  Coverage @ 90%: {result.coverage_90:.4f}")
            click.echo(f"  Sharpness @ 90%: {result.sharpness_90:.6f}")
            click.echo(f"  Winkler @ 90%: {result.winkler_90:.4f}")
            click.echo(f"  Trials: {result.trials}")
            click.echo()
        except Exception as e:
            click.echo(f"  ERROR: {e}", err=True)
            click.echo()

    click.echo(f"\n{'=' * 60}")
    click.echo("Summary - Best Configurations")
    click.echo(f"{'=' * 60}\n")

    for model_name, params in best_configs.items():
        click.echo(f"  {model_name}:")
        for k, v in params.items():
            click.echo(f"    {k}: {v}")
        click.echo()

    if output:
        output_data = {
            "dataset": dataset,
            "target_coverage": target_coverage,
            "n_samples": n_samples,
            "results": [
                {
                    "model": r.model_name,
                    "best_params": r.best_params,
                    "coverage_90": r.coverage_90,
                    "sharpness_90": r.sharpness_90,
                    "winkler_90": r.winkler_90,
                    "trials": r.trials,
                }
                for r in results
            ],
        }
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Tuned parameters saved to: {output_path}")


@cli.command()
@click.argument("dataset", type=str)
@click.option(
    "--cache-dir",
    type=str,
    default=None,
    help="Custom cache directory for HuggingFace datasets",
)
def download_dataset_cmd(dataset: str, cache_dir: str | None) -> None:
    """Download a dataset for offline use.

    Examples:

        # Download a single dataset
        uncertainty-flow download-dataset m4_daily

        # Download exchange rate dataset
        uncertainty-flow download-dataset exchange_rate
    """
    try:
        click.echo(f"Downloading dataset: {dataset}...")
        path = download_dataset(dataset, cache_dir=cache_dir)
        click.echo(f"Dataset saved to: {path}")

        import polars as pl

        df = pl.read_parquet(path)
        click.echo(f"Dataset size: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        click.echo(f"Error downloading dataset: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--domain",
    type=str,
    default=None,
    help="Filter by domain",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Output file for dataset list",
)
def download_all(domain: str | None, output: str | None) -> None:
    """Download all chronos datasets or filter by domain.

    This will download all 67 datasets from autogluon/chronos_datasets.

    Note: This may take a while and requires significant disk space.
    """
    if domain:
        datasets = [ds for ds in AVAILABLE_DATASETS.values() if ds.domain == domain]
        if not datasets:
            click.echo(f"No datasets found for domain '{domain}'.", err=True)
            sys.exit(1)
    else:
        datasets = [
            ds for ds in AVAILABLE_DATASETS.values() if ds.hf_path == "autogluon/chronos_datasets"
        ]

    click.echo(f"Will download {len(datasets)} datasets...")

    for i, ds in enumerate(sorted(datasets, key=lambda x: x.name), 1):
        try:
            click.echo(f"[{i}/{len(datasets)}] Downloading {ds.name}...")
            path = download_dataset(ds.name)
            click.echo(f"  -> {path}")
        except Exception as e:
            click.echo(f"  ERROR: {e}", err=True)

    click.echo("\nDownload complete!")

    if output:
        with open(output, "w") as f:
            for ds in sorted(datasets, key=lambda x: x.name):
                f.write(f"{ds.name}\n")
        click.echo(f"Dataset list saved to: {output}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
