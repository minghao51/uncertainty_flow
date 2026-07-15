"""Side-effect-free first Hamilton nodes for the benchmark lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from random import Random

import polars as pl

from ..contracts.runs import ResolvedRunConfig, RunRequest
from ..contracts.validation import (
    LeakageCheckResult,
    SplitAssignment,
    SplitStrategy,
    ValidationPlan,
)
from ..identity import content_hash
from ..registry import default_metric_registry, default_model_registry

_REDACTED_SECRET = "<redacted>"


def resolved_run_config(run_request: RunRequest) -> ResolvedRunConfig:
    """Resolve a request into the immutable configuration consumed by nodes."""

    if not run_request.models:
        raise ValueError("At least one model is required")
    dataset_id = run_request.dataset.get("id")
    dataset_path = Path(str(dataset_id)) if dataset_id is not None else Path()
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_path.is_absolute()
        or ".." in dataset_path.parts
    ):
        raise ValueError("dataset.id must be a non-empty, relative identifier")
    resolved_models = default_model_registry().resolve_specs(run_request.models)
    metrics = run_request.evaluation.get("metrics", ["coverage", "winkler", "pinball"])
    if (
        not isinstance(metrics, list)
        or not metrics
        or not all(isinstance(metric, str) for metric in metrics)
    ):
        raise ValueError("evaluation.metrics must be a non-empty list of metric names")
    metric_registry = default_metric_registry()
    for metric in metrics:
        metric_registry.get(metric)
    coverage_levels = run_request.evaluation.get("coverage_levels", [0.8, 0.9])
    if not isinstance(coverage_levels, list) or not coverage_levels:
        raise ValueError("evaluation.coverage_levels must be a non-empty list")
    if any(
        isinstance(level, bool) or not isinstance(level, (int, float)) or not 0 < level < 1
        for level in coverage_levels
    ):
        raise ValueError("evaluation.coverage_levels must contain numbers between 0 and 1")
    publication = {
        key: _REDACTED_SECRET if key == "manifest_secret" and value else value
        for key, value in run_request.publication.items()
    }
    resolved_request = run_request.model_copy(
        update={"models": resolved_models, "publication": publication}
    )
    return ResolvedRunConfig(
        request=resolved_request,
        config_hash=content_hash(resolved_request),
        code_version="uncertainty-flow-0.5.0",
    )


def source_dataset(input_frame: pl.DataFrame) -> pl.DataFrame:
    """Provide the source frame as an explicit DAG input."""

    return input_frame


def validation_plan(
    source_dataset: pl.DataFrame,
    resolved_run_config: ResolvedRunConfig,
) -> ValidationPlan:
    """Create a deterministic random or temporal holdout split plan."""

    settings = resolved_run_config.request.validation
    strategy = SplitStrategy(str(settings.get("strategy", SplitStrategy.RANDOM_HOLDOUT)))
    if strategy not in (
        SplitStrategy.RANDOM_HOLDOUT,
        SplitStrategy.TEMPORAL_HOLDOUT,
        SplitStrategy.ROLLING_ORIGIN,
    ):
        raise ValueError(f"Unsupported validation strategy {strategy}")

    test_size = float(settings.get("test_size", 0.2))
    if not 0 < test_size < 1:
        raise ValueError("validation.test_size must be between 0 and 1")
    if len(source_dataset) < 3:
        raise ValueError("Validation requires at least 3 observations")

    id_column = "id" if "id" in source_dataset.columns else None
    observation_ids = (
        [str(value) for value in source_dataset["id"].to_list()]
        if id_column
        else [str(index) for index in range(len(source_dataset))]
    )
    if id_column and source_dataset["id"].null_count() > 0:
        raise ValueError("Validation input observation IDs cannot contain nulls")
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("Validation input contains duplicate observation IDs")

    n_test = max(1, int(len(observation_ids) * test_size))
    n_train = len(observation_ids) - n_test
    if n_train < 2:
        raise ValueError("validation.test_size leaves fewer than two training observations")

    ordered_indices = list(range(len(observation_ids)))
    if strategy == SplitStrategy.RANDOM_HOLDOUT:
        Random(int(settings.get("random_seed", 42))).shuffle(ordered_indices)
    elif strategy == SplitStrategy.TEMPORAL_HOLDOUT or strategy == SplitStrategy.ROLLING_ORIGIN:
        timestamp_column = settings.get("timestamp_column")
        preserve_order = settings.get("preserve_order", False)
        if not isinstance(preserve_order, bool):
            raise ValueError("validation.preserve_order must be a boolean")
        if (
            strategy == SplitStrategy.TEMPORAL_HOLDOUT
            and (
                not isinstance(timestamp_column, str)
                or timestamp_column not in source_dataset.columns
            )
            and not preserve_order
        ):
            raise ValueError(
                "temporal_holdout requires validation.timestamp_column or preserve_order: true"
            )
        if isinstance(timestamp_column, str) and timestamp_column in source_dataset.columns:
            timestamps = source_dataset[timestamp_column].to_list()
            if any(timestamp is None for timestamp in timestamps):
                raise ValueError("validation timestamp column cannot contain nulls")
            ordered_indices.sort(key=lambda index: (timestamps[index], index))

    if strategy == SplitStrategy.ROLLING_ORIGIN:
        n_folds = int(settings.get("n_folds", 3))
        step_size = int(settings.get("step_size", n_test))
        default_min_train = len(ordered_indices) - n_test - step_size * (n_folds - 1)
        min_train_size = int(settings.get("min_train_size", default_min_train))
        if n_folds < 1 or step_size < 1 or min_train_size < 2:
            raise ValueError(
                "rolling_origin requires positive n_folds/step_size and at least two training rows"
            )
        assignments_list: list[SplitAssignment] = []
        for fold in range(n_folds):
            train_end = min_train_size + fold * step_size
            test_end = train_end + n_test
            if test_end > len(ordered_indices):
                raise ValueError(
                    "rolling_origin configuration does not fit the available observations"
                )
            train_indices = set(ordered_indices[:train_end])
            test_indices = set(ordered_indices[train_end:test_end])
            assignments_list.extend(
                SplitAssignment(
                    observation_id=observation_id,
                    split="test" if index in test_indices else "train",
                    fold=fold,
                )
                for index, observation_id in enumerate(observation_ids)
                if index in train_indices or index in test_indices
            )
        assignments = tuple(assignments_list)
    else:
        test_indices = set(ordered_indices[-n_test:])
        assignments = tuple(
            SplitAssignment(
                observation_id=observation_id,
                split="test" if index in test_indices else "train",
            )
            for index, observation_id in enumerate(observation_ids)
        )
    train_ids = {item.observation_id for item in assignments if item.split == "train"}
    test_ids = {item.observation_id for item in assignments if item.split == "test"}
    leakage = LeakageCheckResult(
        passed=(
            strategy == SplitStrategy.ROLLING_ORIGIN
            or (train_ids.isdisjoint(test_ids) and train_ids | test_ids == set(observation_ids))
        ),
        checked_rows=len(assignments),
    )
    plan_id = content_hash(
        {
            "validation": settings,
            "assignments": assignments,
        }
    )
    return ValidationPlan(
        validation_plan_id=plan_id,
        strategy=strategy,
        random_seed=settings.get("random_seed"),
        test_size=test_size,
        calibration_size=settings.get("calibration_size"),
        assignments=assignments,
        leakage_check=leakage,
    )


def validation_timestamp() -> str:
    """Expose a timestamp only for diagnostics, never for identity."""

    return datetime.now(timezone.utc).isoformat()
