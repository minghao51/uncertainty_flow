"""Benchmark flow and model build configs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    dataset_name: str
    n_samples: int = 1000
    confidence_levels: list[float] | None = None
    random_state: int = 42
    auto_tune: bool = True
    target_coverage: float = 0.9
    tune_samples: int = 500
    tune_timeout: int = 120
    test_size: float = 0.2
    tune_size: float = 0.2
    dataset_revision: str | None = None
    hybrid_validation: bool = False
    rolling_origin: bool = False
    rolling_n_splits: int = 5
    rolling_min_train: int = 50
    rolling_horizon: int = 1

    def __post_init__(self) -> None:
        if self.confidence_levels is None:
            self.confidence_levels = [0.8, 0.9, 0.95]


@dataclass
class ModelBuildConfig:
    """Configuration consumed by model adapters."""

    model_name: str
    target_column: str
    horizon: int = 3
    n_estimators: int = 30
    random_state: int = 42
    tuned_params: dict[str, object] | None = None


# Backward-compatible public name retained for callers.
@dataclass
class BenchmarkConfig(RunConfig):
    """Legacy benchmark config retained as public API."""

    horizon: int = 3
    n_estimators: int = 30
    target_column: str | None = None
