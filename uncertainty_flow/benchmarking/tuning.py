"""Hyperparameter tuning for benchmark models."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.metrics import coverage_score, winkler_score
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.utils.exceptions import ConfigurationError
from uncertainty_flow.wrappers import ConformalForecaster, ConformalRegressor

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
    df: pl.DataFrame,
    target: str,
    horizon: int,
    n_estimators: int,
) -> tuple[float, float, float, float]:
    """Tune and evaluate quantile forest."""
    import time

    start = time.time()
    model = QuantileForestForecaster(
        targets=target,
        horizon=horizon,
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(df)
    pred = model.predict(df)
    train_time = time.time() - start

    interval_90 = pred.interval(0.9)
    n_pred = len(interval_90)
    y_true = df[target].to_numpy()[-n_pred:]
    lower = interval_90["lower"].to_numpy()
    upper = interval_90["upper"].to_numpy()

    cov = coverage_score(y_true, lower, upper)
    sharp = float(np.mean(upper - lower))
    wink = winkler_score(y_true, lower, upper, confidence=0.9)

    return cov, sharp, wink, train_time


def tune_conformal_regressor(
    df: pl.DataFrame,
    target: str,
    n_estimators: int,
    calibration_size: float,
) -> tuple[float, float, float, float]:
    """Tune and evaluate conformal regressor."""
    import time

    start = time.time()
    model = ConformalRegressor(
        base_model=GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=42,
        ),
        calibration_size=calibration_size,
        random_state=42,
    )
    model.fit(df, target=target)
    pred = model.predict(df)
    train_time = time.time() - start

    interval_90 = pred.interval(0.9)
    n_pred = len(interval_90)
    y_true = df[target].to_numpy()[-n_pred:]
    lower = interval_90["lower"].to_numpy()
    upper = interval_90["upper"].to_numpy()

    cov = coverage_score(y_true, lower, upper)
    sharp = float(np.mean(upper - lower))
    wink = winkler_score(y_true, lower, upper, confidence=0.9)

    return cov, sharp, wink, train_time


def tune_conformal_forecaster(
    df: pl.DataFrame,
    target: str,
    horizon: int,
    n_estimators: int,
    calibration_size: float,
    lags: int,
) -> tuple[float, float, float, float]:
    """Tune and evaluate conformal forecaster."""
    import time

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
        random_state=42,
    )
    model.fit(df)
    pred = model.predict(df)
    train_time = time.time() - start

    interval_90 = pred.interval(0.9)
    n_pred = len(interval_90)
    y_true = df[target].to_numpy()[-n_pred:]
    lower = interval_90["lower"].to_numpy()
    upper = interval_90["upper"].to_numpy()

    cov = coverage_score(y_true, lower, upper)
    sharp = float(np.mean(upper - lower))
    wink = winkler_score(y_true, lower, upper, confidence=0.9)

    return cov, sharp, wink, train_time


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

    if model_name == "quantile-forest":
        for n_est in search_space.get("n_estimators", [30]):
            for h in search_space.get("horizon", [3]):
                trials += 1
                cov, sharp, wink, train_time = tune_quantile_forest(df, target, h, n_est)
                score = _score_result(cov, sharp, wink, config.target_coverage)
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
            for calib in search_space.get("calibration_size", [0.2]):
                trials += 1
                cov, sharp, wink, train_time = tune_conformal_regressor(df, target, n_est, calib)
                score = _score_result(cov, sharp, wink, config.target_coverage)
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
            for calib in search_space.get("calibration_size", [0.2]):
                for lags in search_space.get("lags", [2]):
                    trials += 1
                    cov, sharp, wink, train_time = tune_conformal_forecaster(
                        df, target, horizon, n_est, calib, lags
                    )
                    score = _score_result(cov, sharp, wink, config.target_coverage)
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

    cov_90 = best_metrics.get("coverage_90", 0)
    sharp_90 = best_metrics.get("sharpness_90", 0)
    wink_90 = best_metrics.get("winkler_90", 0)
    train_time = best_metrics.get("train_time", 0)

    return TuningResult(
        model_name=model_name,
        best_params=best_params,
        best_score=best_score,
        coverage_90=round(cov_90, 4),
        sharpness_90=round(sharp_90, 6),
        winkler_90=round(wink_90, 4),
        train_time_sec=round(train_time, 3),
        trials=trials,
    )


def auto_tune(
    dataset_name: str,
    model_name: str,
    n_samples: int = 1000,
    target_coverage: float = 0.9,
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

    df, _ = load_dataset(dataset_name, n_samples=n_samples)
    target = df.columns[-1]

    config = TuningConfig(target_coverage=target_coverage, n_samples=n_samples)

    return auto_tune_model(
        model_name=model_name,
        df=df,
        target=target,
        horizon=3,
        config=config,
    )
