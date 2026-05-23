"""Benchmark result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResult:
    """Results for a single model on a dataset."""

    model_name: str
    coverage_90: float
    coverage_80: float
    sharpness_90: float
    sharpness_80: float
    winkler_90: float
    winkler_80: float
    pinball_loss: float
    train_time_sec: float
    n_samples: int
    tuned_params: dict[str, Any] = field(default_factory=dict)
    was_tuned: bool = False
    validation_coverage_90: float | None = None
    validation_sharpness_90: float | None = None
    validation_winkler_90: float | None = None
    validation_split_type: str | None = None
    validation_strategy: str | None = None
    validation_n_splits: int | None = None
    test_split_type: str = "out_of_time"


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a dataset."""

    run_id: str
    timestamp: str
    dataset_name: str
    dataset_domain: str
    n_samples: int
    horizon: int
    models: list[ModelResult]
    errors: list[dict[str, str]] = field(default_factory=list)
