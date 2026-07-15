#!/usr/bin/env python3
"""CLI for uncertainty_flow benchmarking."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

from uncertainty_flow import __version__
from uncertainty_flow.benchmarking import (
    AVAILABLE_DATASETS,
    DatasetRegistry,
    TuningResult,
    default_dataset_registry,
    default_model_registry,
)
from uncertainty_flow.benchmarking.contracts import (
    ArtifactRef,
    ArtifactType,
    ReusePolicy,
    RunManifest,
    RunRequest,
)
from uncertainty_flow.benchmarking.contracts.verification import VerificationStatus
from uncertainty_flow.benchmarking.dataflows.vertical import resolved_run_config
from uncertainty_flow.benchmarking.datasets import download_dataset, load_dataset
from uncertainty_flow.benchmarking.driver import available_outputs, build_driver
from uncertainty_flow.benchmarking.evidence import export_evidence
from uncertainty_flow.benchmarking.matrix import ModelMatrixCoordinator
from uncertainty_flow.benchmarking.operations import prune_unverified_runs
from uncertainty_flow.benchmarking.storage import LocalArtifactStore
from uncertainty_flow.benchmarking.tuning import TuningConfig, auto_tune_model
from uncertainty_flow.utils.exceptions import RECOVERABLE_EXCEPTIONS, ConfigurationError

logger = logging.getLogger(__name__)
RECOVERABLE_CLI_EXCEPTIONS = RECOVERABLE_EXCEPTIONS + (
    click.ClickException,
    ConfigurationError,
    ImportError,
    RuntimeError,
    ValueError,
)


def _load_pipeline_request(
    path: Path,
    *,
    storage_root: str | None = None,
    reuse_policy: str | None = None,
) -> RunRequest:
    """Resolve a pipeline request from CLI, environment, file, then defaults."""

    try:
        import yaml
    except ImportError as error:
        raise ConfigurationError(
            "Pipeline configuration requires the benchmarking extra: uv sync --extra benchmarking"
        ) from error
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise click.ClickException("Pipeline config must contain a mapping")
    pipeline_config = dict(payload.get("pipeline", {}))
    dataset = dict(payload.get("dataset", {}))
    dataset.setdefault("provider", "local_parquet")
    validation = dict(payload.get("validation", {}))
    models = tuple(dict(model) for model in payload.get("models", ()))
    storage_config = dict(payload.get("storage", {}))
    resolved_root = storage_root or os.environ.get("UNCERTAINTY_FLOW_PIPELINE_STORAGE_ROOT")
    if resolved_root:
        storage_config["root"] = resolved_root
    resolved_reuse_policy = reuse_policy or os.environ.get("UNCERTAINTY_FLOW_PIPELINE_REUSE_POLICY")
    return RunRequest(
        mode=str(pipeline_config.get("mode", "benchmark")),
        dataset=dataset,
        validation=validation,
        models=models,
        evaluation=dict(payload.get("evaluation", {})),
        storage=storage_config,
        publication=dict(payload.get("publication", {})),
        reuse_policy=ReusePolicy(
            resolved_reuse_policy or pipeline_config.get("reuse_policy", "reuse_verified")
        ),
        fail_fast=bool(pipeline_config.get("fail_fast", False)),
    )


@click.group()
@click.version_option(version=__version__, prog_name="uncertainty-flow")
def cli() -> None:
    """uncertainty-flow: Probabilistic forecasting and uncertainty quantification."""
    pass


@cli.group()
def pipeline() -> None:
    """Plan, execute, and verify the Hamilton benchmark pipeline."""


@pipeline.command("plan")
@click.option(
    "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--storage-root", type=str, default=None)
@click.option(
    "--reuse-policy",
    type=click.Choice(["reuse_verified", "fail_if_exists", "rerun"]),
    default=None,
)
def pipeline_plan(config: Path, storage_root: str | None, reuse_policy: str | None) -> None:
    """Validate configuration and DAG construction without side effects."""

    try:
        request = _load_pipeline_request(
            config, storage_root=storage_root, reuse_policy=reuse_policy
        )
        resolved_run_config(request)
        build_driver()
    except RECOVERABLE_CLI_EXCEPTIONS as error:
        raise click.ClickException(str(error)) from error
    click.echo("Pipeline plan valid")
    click.echo(f"  Mode: {request.mode}")
    click.echo(f"  Outputs: {', '.join(available_outputs())}")
    click.echo("  Side effects: none")


@pipeline.command("run")
@click.option(
    "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--storage-root", type=str, default=None)
@click.option(
    "--reuse-policy",
    type=click.Choice(["reuse_verified", "fail_if_exists", "rerun"]),
    default=None,
)
def pipeline_run(config: Path, storage_root: str | None, reuse_policy: str | None) -> None:
    """Execute the initial local conformal-regressor pipeline branch."""

    try:
        request = _load_pipeline_request(
            config, storage_root=storage_root, reuse_policy=reuse_policy
        )
        uri = request.dataset.get("uri")
        if not isinstance(uri, str) or not uri:
            raise click.ClickException("dataset.uri is required for pipeline run")
        provider = request.dataset.get("provider", "local_parquet")
        if not isinstance(provider, str) or not provider:
            raise click.ClickException("dataset.provider must be a non-empty string")
        dataset_registry = default_dataset_registry()
        frame = dataset_registry.load(provider, uri)
        dataset_version = dataset_registry.version(provider)
        request = request.model_copy(
            update={
                "dataset": {
                    **request.dataset,
                    "adapter_version": dataset_version,
                }
            }
        )
        storage_root = request.storage.get("root", "data")
        matrix_result = ModelMatrixCoordinator(storage_root=str(storage_root)).run_with_lock(
            request, frame
        )
        manifest = matrix_result.manifest
        verified = matrix_result.verification.passed
    except RECOVERABLE_CLI_EXCEPTIONS as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Run: {manifest.identity.run_id}")
    click.echo(f"Status: {manifest.status.value}")
    click.echo(f"Verified: {verified}")


def _read_manifest(root: Path, run_id: str) -> tuple[LocalArtifactStore, RunManifest]:
    store = LocalArtifactStore(root)
    ref = ArtifactRef(
        artifact_type=ArtifactType.MANIFEST,
        path=f"04_platinum/runs/{run_id}/manifest.json",
        schema_version="v1",
    )
    if not store.exists(ref):
        raise click.ClickException(f"Run not found: {run_id}")
    return store, RunManifest.model_validate(store.read_json(ref))


@pipeline.command("verify")
@click.argument("run_id")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default="data")
def pipeline_verify(run_id: str, root: Path) -> None:
    """Verify every checksummed artifact referenced by one run manifest."""

    try:
        store, manifest = _read_manifest(root, run_id)
        invalid = [
            ref.path
            for ref in manifest.artifact_refs
            if store.verify(ref).status != VerificationStatus.PASSED
        ]
    except RECOVERABLE_CLI_EXCEPTIONS as error:
        raise click.ClickException(str(error)) from error
    if invalid or not manifest.verification_passed:
        raise click.ClickException(
            f"Run {run_id} is unverified" + (f": {', '.join(invalid)}" if invalid else "")
        )
    click.echo(f"Run {run_id} verified")


@pipeline.command("lineage")
@click.argument("run_id")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default="data")
def pipeline_lineage(run_id: str, root: Path) -> None:
    """Print the immutable artifacts referenced by one run."""

    try:
        _, manifest = _read_manifest(root, run_id)
    except RECOVERABLE_CLI_EXCEPTIONS as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Run: {manifest.identity.run_id}")
    for ref in manifest.artifact_refs:
        click.echo(ref.path)


@pipeline.command("list-runs")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default="data")
def pipeline_list_runs(root: Path) -> None:
    """List persisted Platinum runs and their verification state."""

    manifests = sorted((root / "04_platinum" / "runs").glob("*/manifest.json"))
    for path in manifests:
        try:
            manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except RECOVERABLE_CLI_EXCEPTIONS as error:
            raise click.ClickException(f"Invalid manifest {path}: {error}") from error
        click.echo(
            f"{manifest.identity.run_id}\t{manifest.status.value}\t"
            f"verified={manifest.verification_passed}"
        )


@pipeline.command("export-site")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default="data")
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True)
def pipeline_export_site(root: Path, output: Path) -> None:
    """Export verified Platinum summaries for the evidence site."""

    try:
        index = export_evidence(root, output)
    except RECOVERABLE_CLI_EXCEPTIONS as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Evidence export complete: {len(index.partitions)} partition(s)")


@pipeline.command("gc")
@click.option("--root", type=click.Path(file_okay=False, path_type=Path), default="data")
@click.option("--apply", "apply_changes", is_flag=True, help="Delete failed/incomplete runs.")
def pipeline_gc(root: Path, apply_changes: bool) -> None:
    """Preview or remove only failed/incomplete Platinum runs."""

    removed = prune_unverified_runs(root, dry_run=not apply_changes)
    action = "Removed" if apply_changes else "Would remove"
    click.echo(f"{action} {len(removed)} unverified run(s)")


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
    "--samples",
    "-s",
    type=int,
    default=100,
    help="Number of samples to use from dataset (default: 100)",
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
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
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
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Continue running other models when a model fails and return partial results.",
)
@click.option(
    "--test-size",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=0.2,
    help="Fraction of data to hold out for testing (default: 0.2)",
)
@click.option(
    "--dataset-revision",
    type=str,
    default=None,
    help="Pinned HuggingFace dataset revision (commit hash).",
)
@click.option(
    "--hybrid-validation/--no-hybrid-validation",
    default=False,
    help="Use hybrid validation (outer split + inner out-of-sample CV) during auto-tuning.",
)
def benchmark(
    dataset: str,
    model: str,
    samples: int,
    horizon: int,
    n_estimators: int,
    target: str | None,
    auto_tune: bool,
    target_coverage: float,
    tune_samples: int,
    output: str | None,
    json_only: bool,
    csv_only: bool,
    allow_partial: bool,
    test_size: float,
    dataset_revision: str | None,
    hybrid_validation: bool,
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
    if model == "all":
        model_names = ["quantile-forest", "conformal-regressor", "conformal-forecaster"]
    else:
        model_names = [m.strip() for m in model.split(",")]
        valid_models = set(default_model_registry().names())
        for m in model_names:
            if m not in valid_models:
                click.echo(
                    f"Error: Unknown model '{m}'. Valid options: {', '.join(valid_models)}",
                    err=True,
                )
                sys.exit(1)
    if json_only and csv_only:
        raise click.UsageError("--json-only and --csv-only are mutually exclusive")

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Benchmark: {dataset}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Samples: {samples}")
    click.echo(f"  Horizon: {horizon}")
    click.echo(f"  Estimators: {n_estimators}")
    if target:
        click.echo(f"  Target: {target}")
    click.echo(f"  Models: {model}")
    click.echo()

    try:
        click.echo("Loading dataset...")
        loaded_frame, dataset_info = load_dataset(
            dataset,
            n_samples=samples,
            revision=dataset_revision,
        )
        target_column = target or dataset_info.default_target
        if target_column not in loaded_frame.columns:
            raise ConfigurationError(
                f"Target column {target_column!r} is not present in dataset {dataset!r}"
            )
        dataset_registry = DatasetRegistry()
        dataset_registry.register(
            "benchmark_dataset",
            lambda _uri: loaded_frame,
            version=f"benchmark-v1:{dataset_revision or 'resolved'}",
        )
        frame = dataset_registry.load("benchmark_dataset", dataset)
        tuned_parameters: dict[str, dict[str, object]] = {}
        if auto_tune:
            for model_name in model_names:
                tuned_parameters[model_name] = auto_tune_model(
                    model_name,
                    frame,
                    target_column,
                    horizon,
                    TuningConfig(
                        target_coverage=target_coverage,
                        n_samples=tune_samples,
                        hybrid_validation=hybrid_validation,
                    ),
                ).best_params
        request = RunRequest(
            dataset={
                "id": dataset,
                "provider": "benchmark_dataset",
                "uri": dataset,
                "target": target_column,
                "domain": dataset_info.domain,
                "adapter_version": dataset_registry.version("benchmark_dataset"),
                "source_revision": dataset_revision,
            },
            validation={
                "strategy": "temporal_holdout",
                "test_size": test_size,
                "preserve_order": True,
            },
            models=tuple(
                {
                    "id": model_name,
                    "provider": model_name,
                    "required": not allow_partial,
                    "parameters": {
                        "horizon": horizon,
                        "n_estimators": n_estimators,
                        "random_state": 42,
                        **tuned_parameters.get(model_name, {}),
                    },
                }
                for model_name in model_names
            ),
            evaluation={
                "metrics": [
                    "coverage",
                    "sharpness",
                    "winkler",
                    "pinball",
                    "crps",
                    "mae",
                    "rmse",
                    "calibration_error",
                ],
                "coverage_levels": [0.8, 0.9],
            },
            storage={"provider": "local", "root": "data"},
        )
        result = ModelMatrixCoordinator(storage_root="data").run_with_lock(request, frame)
        click.echo(f"  Loaded: {len(frame):,} rows, {len(frame.columns)} columns")
        click.echo(f"  Domain: {dataset_info.domain}")
        click.echo(f"  Target: {target_column}")
        click.echo()

        click.echo(f"\n{'=' * 60}")
        click.echo("Results")
        click.echo(f"{'=' * 60}\n")

        for model_result in result.model_results:
            click.echo(f"  {model_result.model_id} [{model_result.status.value}]:")
            for metric_name, metric_value in sorted(model_result.metrics.items()):
                click.echo(f"    {metric_name}: {metric_value:.4f}")
            click.echo(f"    Train time: {model_result.train_time_sec:.3f}s")
            click.echo()

        failed_models = [item for item in result.model_results if item.error is not None]
        if failed_models:
            click.echo("Model failures:")
            for model_result in failed_models:
                click.echo(f"  - {model_result.model_id}: {model_result.error}")
            click.echo()

        json_path: Path
        csv_path: Path
        if output:
            output_path = Path(output)
            json_path = output_path.with_suffix(".json")
            csv_path = output_path.with_suffix(".csv")

        else:
            default_json = Path("benchmark_results.json")
            default_csv = Path("benchmark_results.csv")
            json_path = default_json
            csv_path = default_csv

        if not csv_only:
            json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            click.echo(f"JSON results saved to: {json_path}")
        if not json_only:
            rows = [
                {
                    "model": model_result.model_id,
                    "provider": model_result.provider,
                    "status": model_result.status.value,
                    "train_time_sec": model_result.train_time_sec,
                    "evaluation_row_count": model_result.evaluation_row_count,
                    **model_result.metrics,
                }
                for model_result in result.model_results
            ]
            import polars as pl

            pl.DataFrame(rows).write_csv(csv_path)
            click.echo(f"CSV results saved to: {csv_path}")

    except (KeyboardInterrupt, SystemExit):
        raise
    except RECOVERABLE_CLI_EXCEPTIONS as e:
        logger.exception("Benchmark command failed: %s", e)
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
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
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
@click.option(
    "--dataset-revision",
    type=str,
    default=None,
    help="Pinned HuggingFace dataset revision (commit hash).",
)
def tune(
    dataset: str,
    model: str,
    n_samples: int,
    target_coverage: float,
    output: str | None,
    dataset_revision: str | None,
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

    try:
        loaded_frame, dataset_info = load_dataset(
            dataset,
            n_samples=n_samples,
            revision=dataset_revision,
        )
        target_column = dataset_info.default_target
        dataset_registry = DatasetRegistry()
        dataset_registry.register(
            "benchmark_dataset",
            lambda _uri: loaded_frame,
            version=f"benchmark-v1:{dataset_revision or 'resolved'}",
        )
        frame = dataset_registry.load("benchmark_dataset", dataset)
    except RECOVERABLE_CLI_EXCEPTIONS as error:
        raise click.ClickException(str(error)) from error

    for model_name in model_names:
        try:
            click.echo(f"[{model_name}]")
            result = auto_tune_model(
                model_name=model_name,
                df=frame,
                target=target_column,
                horizon=3,
                config=TuningConfig(
                    target_coverage=target_coverage,
                    n_samples=n_samples,
                ),
            )
            results.append(result)
            best_configs[model_name] = result.best_params

            click.echo(f"  Best params: {result.best_params}")
            click.echo(f"  Coverage @ 90%: {result.coverage_90:.4f}")
            click.echo(f"  Sharpness @ 90%: {result.sharpness_90:.6f}")
            click.echo(f"  Winkler @ 90%: {result.winkler_90:.4f}")
            click.echo(f"  Trials: {result.trials}")
            click.echo()
        except (KeyboardInterrupt, SystemExit):
            raise
        except RECOVERABLE_CLI_EXCEPTIONS as e:
            logger.exception("Tune command model '%s' failed: %s", model_name, e)
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
@click.option(
    "--dataset-revision",
    type=str,
    default=None,
    help="Pinned HuggingFace dataset revision (commit hash).",
)
def download_dataset_cmd(
    dataset: str,
    cache_dir: str | None,
    dataset_revision: str | None,
) -> None:
    """Download a dataset for offline use.

    Examples:

        # Download a single dataset
        uncertainty-flow download-dataset m4_daily

        # Download exchange rate dataset
        uncertainty-flow download-dataset exchange_rate
    """
    try:
        click.echo(f"Downloading dataset: {dataset}...")
        path = download_dataset(
            dataset,
            cache_dir=cache_dir,
            revision=dataset_revision,
        )
        click.echo(f"Dataset saved to: {path}")

        import polars as pl

        df = pl.read_parquet(path)
        click.echo(f"Dataset size: {len(df):,} rows, {len(df.columns)} columns")
    except (KeyboardInterrupt, SystemExit):
        raise
    except RECOVERABLE_CLI_EXCEPTIONS as e:
        logger.exception("download-dataset command failed for '%s': %s", dataset, e)
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except RECOVERABLE_CLI_EXCEPTIONS as e:
            logger.exception("download-all failed for dataset '%s': %s", ds.name, e)
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
