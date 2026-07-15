"""Hyperparameter tuning for benchmark models."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.metrics import coverage_score, winkler_score
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.utils.auto_tuning import valid_calibration_candidates
from uncertainty_flow.utils.exceptions import ConfigurationError
from uncertainty_flow.utils.polars_bridge import to_numpy_series
from uncertainty_flow.utils.split import select_validation_plan
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor

logger = logging.getLogger(__name__)

SEARCH_SPACE: dict[str, dict[str, list[Any]]] = {
    "quantile-forest": {
        "n_estimators": [20, 30, 50],
        "horizon": [2, 3, 5],
    },
    "conformal-regressor": {
        "n_estimators": [20, 30, 50],
        "calibration_size": [0.15, 0.20, 0.25, 0.30],
    },
    "conformal-forecaster": {
        "n_estimators": [20, 30, 50],
        "calibration_size": [0.15, 0.20, 0.25],
        "lags": [1, 2, 3],
    },
}


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning.

    Attributes:
        target_coverage: Target coverage level (default: 0.9)
        n_samples: Number of samples for tuning (default: 500)
        timeout: Maximum seconds per model (default: 120)
    """

    target_coverage: float = 0.9
    n_samples: int = 500
    timeout: int = 120
    hybrid_validation: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.target_coverage < 1:
            raise ValueError("target_coverage must be between 0 and 1")
        if self.n_samples < 3:
            raise ValueError("n_samples must be at least 3")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    model_name: str
    best_params: dict[str, Any]
    best_score: float
    coverage_90: float
    sharpness_90: float
    winkler_90: float
    train_time_sec: float
    trials: int
    validation_strategy: str = "unknown"
    validation_split_type: str = "unknown"
    validation_n_splits: int = 1


def _score_result(
    coverage: float,
    sharpness: float,
    winkler: float,
    target_coverage: float = 0.9,
) -> float:
    """Score a result based on coverage calibration and sharpness.

    Lower is better. Penalizes both under and over coverage.
    """
    coverage_error = abs(coverage - target_coverage)
    if coverage_error > 0.15:
        coverage_error = coverage_error * 10
    return winkler + coverage_error * 0.5 + sharpness * 0.1


def tune_quantile_forest(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    target: str,
    horizon: int,
    n_estimators: int,
) -> tuple[float, float, float, float]:
    """Tune and evaluate quantile forest."""
    start = time.time()
    min_calibration_size = min(0.5, max(0.25, 20 / max(1, len(train_df)) + 0.01))
    model = QuantileForestForecaster(
        targets=target,
        horizon=horizon,
        n_estimators=n_estimators,
        calibration_size=min_calibration_size,
        auto_tune=False,
        random_state=42,
    )
    model.fit(train_df)
    pred = model.predict(val_df)
    train_time = time.time() - start

    interval_90 = pred.interval(0.9)
    n_pred = len(interval_90)
    y_true = to_numpy_series(val_df[target])[-n_pred:]
    lower = to_numpy_series(interval_90["lower"])
    upper = to_numpy_series(interval_90["upper"])

    cov = coverage_score(y_true, lower, upper)
    sharp = float(np.mean(upper - lower))
    wink = winkler_score(y_true, lower, upper, confidence=0.9)

    return cov, sharp, wink, train_time


def tune_conformal_regressor(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    target: str,
    n_estimators: int,
    calibration_size: float,
) -> tuple[float, float, float, float]:
    """Tune and evaluate conformal regressor."""
    start = time.time()
    model = ConformalRegressor(
        base_model=GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=42,
        ),
        calibration_size=calibration_size,
        auto_tune=False,
        random_state=42,
    )
    model.fit(train_df, target=target)
    pred = model.predict(val_df)
    train_time = time.time() - start

    interval_90 = pred.interval(0.9)
    n_pred = len(interval_90)
    y_true = to_numpy_series(val_df[target])[-n_pred:]
    lower = to_numpy_series(interval_90["lower"])
    upper = to_numpy_series(interval_90["upper"])

    cov = coverage_score(y_true, lower, upper)
    sharp = float(np.mean(upper - lower))
    wink = winkler_score(y_true, lower, upper, confidence=0.9)

    return cov, sharp, wink, train_time


def tune_conformal_forecaster(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    target: str,
    horizon: int,
    n_estimators: int,
    calibration_size: float,
    lags: int,
) -> tuple[float, float, float, float]:
    """Tune and evaluate conformal forecaster."""
    start = time.time()
    model = ConformalForecaster(
        base_model=GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=42,
        ),
        horizon=horizon,
        targets=target,
        lags=lags,
        calibration_size=calibration_size,
        auto_tune=False,
        random_state=42,
    )
    model.fit(train_df)
    pred = model.predict(val_df)
    train_time = time.time() - start

    interval_90 = pred.interval(0.9)
    n_pred = len(interval_90)
    y_true = to_numpy_series(val_df[target])[-n_pred:]
    lower = to_numpy_series(interval_90["lower"])
    upper = to_numpy_series(interval_90["upper"])

    cov = coverage_score(y_true, lower, upper)
    sharp = float(np.mean(upper - lower))
    wink = winkler_score(y_true, lower, upper, confidence=0.9)

    return cov, sharp, wink, train_time


def _evaluate_over_splits(
    evaluation_splits: list[tuple[pl.DataFrame, pl.DataFrame]],
    model_name: str,
    tune_fn: Callable[[pl.DataFrame, pl.DataFrame], tuple[float, float, float, float]],
    target_coverage: float,
) -> tuple[float, float, float, float, float]:
    """Run *tune_fn* over all evaluation splits, return averaged metrics and score."""
    split_cov: list[float] = []
    split_sharp: list[float] = []
    split_wink: list[float] = []
    split_train_time: list[float] = []
    split_scores: list[float] = []

    for split_idx, (split_train_df, split_val_df) in enumerate(evaluation_splits, start=1):
        logger.debug(
            "tuning_candidate_split model=%s split=%d train_rows=%d val_rows=%d",
            model_name,
            split_idx,
            len(split_train_df),
            len(split_val_df),
        )
        cov, sharp, wink, train_time = tune_fn(split_train_df, split_val_df)
        split_cov.append(cov)
        split_sharp.append(sharp)
        split_wink.append(wink)
        split_train_time.append(train_time)
        split_scores.append(_score_result(cov, sharp, wink, target_coverage))

    return (
        float(np.mean(split_cov)),
        float(np.mean(split_sharp)),
        float(np.mean(split_wink)),
        float(np.mean(split_train_time)),
        float(np.mean(split_scores)),
    )


def auto_tune_model(
    model_name: str,
    df: pl.DataFrame,
    target: str,
    horizon: int,
    config: TuningConfig | None = None,
) -> TuningResult:
    """Automatically tune hyperparameters for a model.

    Args:
        model_name: Name of the model to tune
        df: Polars DataFrame with data
        target: Target column name
        horizon: Forecast horizon
        config: Tuning configuration

    Returns:
        TuningResult with best parameters and scores
    """
    if config is None:
        config = TuningConfig()

    search_space = SEARCH_SPACE.get(model_name, {})
    if not search_space:
        raise ConfigurationError(f"Unknown model: {model_name}")

    best_score = float("inf")
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, float] = {}

    param_combinations = 1
    for values in search_space.values():
        param_combinations *= len(values)

    trials = 0

    task_type: Literal["tabular", "time_series"] = (
        "time_series" if model_name in {"quantile-forest", "conformal-forecaster"} else "tabular"
    )
    if config.n_samples < len(df):
        df = df.tail(config.n_samples) if task_type == "time_series" else df.head(config.n_samples)
    deadline = time.monotonic() + config.timeout
    validation_plan = select_validation_plan(
        df,
        task_type=task_type,
        random_state=42,
        holdout_fraction=0.2,
        hybrid_mode=config.hybrid_validation,
    )
    train_df, val_df = validation_plan.outer_split
    evaluation_splits = (
        validation_plan.inner_splits
        if validation_plan.inner_splits
        else [validation_plan.outer_split]
    )

    logger.info(
        "tuning_validation_plan model=%s strategy=%s reason=%s eval_splits=%d",
        model_name,
        validation_plan.metadata.strategy_name,
        validation_plan.metadata.reason,
        len(evaluation_splits),
    )

    if model_name == "quantile-forest":
        for n_est in search_space.get("n_estimators", [30]):
            for h in search_space.get("horizon", [3]):
                if time.monotonic() >= deadline:
                    break
                trials += 1
                cov, sharp, wink, train_time, score = _evaluate_over_splits(
                    evaluation_splits,
                    model_name,
                    lambda tr, vl: tune_quantile_forest(tr, vl, target, h, n_est),
                    config.target_coverage,
                )
                if score < best_score:
                    best_score = score
                    best_params = {"n_estimators": n_est, "horizon": h}
                    best_metrics = {
                        "coverage_90": cov,
                        "sharpness_90": sharp,
                        "winkler_90": wink,
                        "train_time": train_time,
                    }

    elif model_name == "conformal-regressor":
        for n_est in search_space.get("n_estimators", [30]):
            min_train_rows = min(len(split_train_df) for split_train_df, _ in evaluation_splits)
            calib_candidates = valid_calibration_candidates(
                min_train_rows,
                0.2,
                search_space.get("calibration_size", [0.2]),
            )
            for calib in calib_candidates:
                if time.monotonic() >= deadline:
                    break
                trials += 1
                cov, sharp, wink, train_time, score = _evaluate_over_splits(
                    evaluation_splits,
                    model_name,
                    lambda tr, vl: tune_conformal_regressor(tr, vl, target, n_est, calib),
                    config.target_coverage,
                )
                if score < best_score:
                    best_score = score
                    best_params = {
                        "n_estimators": n_est,
                        "calibration_size": calib,
                    }
                    best_metrics = {
                        "coverage_90": cov,
                        "sharpness_90": sharp,
                        "winkler_90": wink,
                        "train_time": train_time,
                    }

    elif model_name == "conformal-forecaster":
        for n_est in search_space.get("n_estimators", [30]):
            min_train_rows = min(len(split_train_df) for split_train_df, _ in evaluation_splits)
            calib_candidates = valid_calibration_candidates(
                min_train_rows,
                0.2,
                search_space.get("calibration_size", [0.2]),
            )
            for calib in calib_candidates:
                for lags in search_space.get("lags", [2]):
                    if time.monotonic() >= deadline:
                        break
                    trials += 1
                    cov, sharp, wink, train_time, score = _evaluate_over_splits(
                        evaluation_splits,
                        model_name,
                        lambda tr, vl: tune_conformal_forecaster(
                            tr, vl, target, horizon, n_est, calib, lags
                        ),
                        config.target_coverage,
                    )
                    if score < best_score:
                        best_score = score
                        best_params = {
                            "n_estimators": n_est,
                            "calibration_size": calib,
                            "lags": lags,
                        }
                        best_metrics = {
                            "coverage_90": cov,
                            "sharpness_90": sharp,
                            "winkler_90": wink,
                            "train_time": train_time,
                        }

    if not best_metrics:
        raise TimeoutError(f"Tuning {model_name!r} timed out before completing a trial")
    cov_90 = best_metrics["coverage_90"]
    sharp_90 = best_metrics["sharpness_90"]
    wink_90 = best_metrics["winkler_90"]
    train_time = best_metrics["train_time"]
    if validation_plan.metadata.task_type == "time_series":
        validation_split_type = (
            "out_of_time_plus_out_of_sample"
            if validation_plan.metadata.hybrid_mode
            else "out_of_time"
        )
    else:
        validation_split_type = "out_of_sample"

    return TuningResult(
        model_name=model_name,
        best_params=best_params,
        best_score=best_score,
        coverage_90=round(cov_90, 4),
        sharpness_90=round(sharp_90, 6),
        winkler_90=round(wink_90, 4),
        train_time_sec=round(train_time, 3),
        trials=trials,
        validation_strategy=validation_plan.metadata.strategy_name,
        validation_split_type=validation_split_type,
        validation_n_splits=len(evaluation_splits),
    )


def auto_tune(
    dataset_name: str,
    model_name: str,
    n_samples: int = 1000,
    target_coverage: float = 0.9,
    dataset_revision: str | None = None,
) -> TuningResult:
    """Automatically tune hyperparameters for a model on a dataset.

    This is a convenience function that loads the dataset and runs tuning.

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model to tune
        n_samples: Number of samples to use for tuning
        target_coverage: Target coverage level

    Returns:
        TuningResult with best parameters and scores
    """
    from .datasets import load_dataset

    df, _ = load_dataset(
        dataset_name,
        n_samples=n_samples,
        revision=dataset_revision,
    )
    target = df.columns[-1]

    config = TuningConfig(target_coverage=target_coverage, n_samples=n_samples)

    return auto_tune_model(
        model_name=model_name,
        df=df,
        target=target,
        horizon=3,
        config=config,
    )
