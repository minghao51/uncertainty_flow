"""Benchmark model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl

from uncertainty_flow.core.distribution import DistributionPrediction


class BenchmarkModel(Protocol):
    """Interface for benchmark model adapters."""

    train_time: float

    def fit(self, df: pl.DataFrame, target: str) -> None:
        """Fit model on training data."""

    def predict(self, df: pl.DataFrame) -> DistributionPrediction:
        """Predict distributions on evaluation data."""


@dataclass(frozen=True)
class ModelBuildConfig:
    """Resolved configuration consumed by supported model providers."""

    model_name: str
    target_column: str
    horizon: int = 3
    n_estimators: int = 30
    random_state: int = 42
    tuned_params: dict[str, object] | None = None


@dataclass(frozen=True)
class ValidationMetadata:
    """Validation metrics captured during tuning."""

    coverage_90: float | None = None
    sharpness_90: float | None = None
    winkler_90: float | None = None
    split_type: str | None = None
    strategy: str | None = None
    n_splits: int | None = None
