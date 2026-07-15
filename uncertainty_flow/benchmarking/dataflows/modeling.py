"""Hamilton nodes for the first conformal-regressor benchmark branch."""

from __future__ import annotations

import polars as pl

from uncertainty_flow.benchmarking.contracts.runs import ResolvedRunConfig
from uncertainty_flow.benchmarking.contracts.validation import ValidationPlan
from uncertainty_flow.benchmarking.model_contracts import BenchmarkModel, ModelBuildConfig
from uncertainty_flow.benchmarking.registry import default_metric_registry, default_model_registry
from uncertainty_flow.core.distribution import DistributionPrediction


def _model_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove pipeline-only identity columns before model adaptation."""

    columns = [column for column in ("_split", "id") if column in frame.columns]
    return frame.drop(columns)


def gold_dataset(source_dataset: pl.DataFrame, validation_plan: ValidationPlan) -> pl.DataFrame:
    """Attach persisted split membership to the experiment-ready frame."""

    if validation_plan.strategy.value == "rolling_origin":
        ids = (
            [str(value) for value in source_dataset["id"].to_list()]
            if "id" in source_dataset.columns
            else [str(index) for index in range(len(source_dataset))]
        )
        indexed = source_dataset.with_columns(pl.Series("__uf_row_id", ids))
        parts: list[pl.DataFrame] = []
        for assignment in validation_plan.assignments:
            parts.append(
                indexed.filter(pl.col("__uf_row_id") == assignment.observation_id).with_columns(
                    pl.lit(assignment.split).alias("_split"),
                    pl.lit(assignment.fold).alias("_fold"),
                )
            )
        if not parts:
            raise ValueError("Rolling-origin validation plan contains no fold rows")
        return pl.concat(parts, how="vertical").drop("__uf_row_id")

    split_by_id = {item.observation_id: item.split for item in validation_plan.assignments}
    ids = (
        [str(value) for value in source_dataset["id"].to_list()]
        if "id" in source_dataset.columns
        else [str(index) for index in range(len(source_dataset))]
    )
    return source_dataset.with_columns(pl.Series("_split", [split_by_id[value] for value in ids]))


def train_dataset(gold_dataset: pl.DataFrame) -> pl.DataFrame:
    """Select only persisted training membership."""

    return gold_dataset.filter(pl.col("_split") == "train")


def test_dataset(gold_dataset: pl.DataFrame) -> pl.DataFrame:
    """Select only persisted test membership."""

    return gold_dataset.filter(pl.col("_split") == "test")


def target_column(resolved_run_config: ResolvedRunConfig) -> str:
    """Resolve the target column from the dataset request."""

    target = resolved_run_config.request.dataset.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("dataset.target is required for the initial model branch")
    return target


def fitted_model(
    train_dataset: pl.DataFrame,
    target_column: str,
    resolved_run_config: ResolvedRunConfig,
    validation_plan: ValidationPlan,
) -> BenchmarkModel:
    """Build and fit the configured initial benchmark provider."""

    model_specs = resolved_run_config.request.models
    if len(model_specs) != 1:
        raise ValueError("Initial vertical slice requires exactly one model")
    model_name = model_specs[0].get("provider", model_specs[0].get("id"))
    if model_name != "conformal-regressor":
        raise ValueError(f"Initial vertical slice does not support model {model_name!r}")
    parameters = dict(model_specs[0].get("parameters", {}))
    if validation_plan.calibration_size is not None:
        parameters.setdefault("calibration_size", validation_plan.calibration_size)
    provider = default_model_registry().get(model_name)
    model = provider.build(
        ModelBuildConfig(
            model_name=model_name,
            target_column=target_column,
            horizon=int(parameters.get("horizon", 1)),
            n_estimators=int(parameters.get("n_estimators", 30)),
            random_state=int(parameters.get("random_state", 42)),
            tuned_params=parameters,
        )
    )
    model.fit(_model_frame(train_dataset), target_column)
    return model


def distribution_predictions(
    fitted_model: BenchmarkModel,
    test_dataset: pl.DataFrame,
) -> DistributionPrediction:
    """Generate predictions for the persisted test population."""

    return fitted_model.predict(_model_frame(test_dataset))


def metric_results(
    distribution_predictions: DistributionPrediction,
    test_dataset: pl.DataFrame,
    target_column: str,
    resolved_run_config: ResolvedRunConfig,
) -> dict[str, float]:
    """Compute the required initial probabilistic metrics."""

    return evaluate_metrics(
        distribution_predictions,
        test_dataset[target_column],
        resolved_run_config.request.evaluation,
    )


def evaluate_metrics(
    prediction: DistributionPrediction,
    y_true: pl.Series,
    evaluation: dict[str, object],
) -> dict[str, float]:
    """Evaluate exactly the requested univariate metrics and coverage levels."""

    metrics = evaluation.get("metrics", ["coverage", "winkler", "pinball"])
    if (
        not isinstance(metrics, list)
        or not metrics
        or not all(isinstance(metric, str) for metric in metrics)
    ):
        raise ValueError("evaluation.metrics must be a non-empty list of metric names")
    levels = evaluation.get("coverage_levels", [0.8, 0.9])
    if not isinstance(levels, list) or not levels:
        raise ValueError("evaluation.coverage_levels must be a non-empty list")
    coverage_levels = tuple(float(level) for level in levels)
    if any(not 0 < level < 1 for level in coverage_levels):
        raise ValueError("evaluation.coverage_levels must be between 0 and 1")

    results: dict[str, float] = {}
    metric_registry = default_metric_registry()
    for metric in metrics:
        spec = metric_registry.get(metric)
        if spec.per_level:
            for level in coverage_levels:
                key = f"{metric}_{int(level * 100)}"
                results[key] = metric_registry.evaluate(metric, prediction, y_true, level)
        else:
            results[metric] = metric_registry.evaluate(metric, prediction, y_true)
    return results
