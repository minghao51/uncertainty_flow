#!/usr/bin/env python3
"""Reusable utilities for benchmarking uncertainty_flow models.

Provides:
- Multi-iteration timing with warmup
- Statistical aggregation (mean, std, min, max, median)
- Memory measurement helpers
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TimingStats:
    name: str
    values: list[float] = field(default_factory=list)
    unit: str = "s"

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    def summary(self) -> str:
        return (
            f"{self.name}: {self.mean:.4f} ± {self.std:.4f} {self.unit} "
            f"[min={self.min:.4f}, max={self.max:.4f}, median={self.median:.4f}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "median": round(self.median, 6),
            "n_runs": len(self.values),
            "unit": self.unit,
        }


def measure_time(
    fn: Callable[..., Any],
    *args: Any,
    n_iterations: int = 5,
    n_warmup: int = 1,
    **kwargs: Any,
) -> tuple[Any, TimingStats]:
    """Run *fn* multiple times with warmup and return (last_result, TimingStats).

    The warmup runs are discarded so JIT / caching effects don't inflate
    the measured times.
    """
    result = None
    timings: list[float] = []

    for i in range(n_warmup + n_iterations):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if i >= n_warmup:
            timings.append(elapsed)

    stats = TimingStats(name=getattr(fn, "__name__", "unnamed"), values=timings)
    return result, stats


def measure_memory_mb() -> float:
    """Return current RSS in MB (best-effort, cross-platform)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


@dataclass
class BenchmarkScenario:
    name: str
    model_name: str
    dataset: str
    n_samples: int
    horizon: int
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model_name,
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            "horizon": self.horizon,
            "extra_params": self.extra_params,
        }


@dataclass
class BenchmarkRunResult:
    scenario: BenchmarkScenario
    timing: TimingStats
    metrics: dict[str, float]
    memory_delta_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scenario.to_dict(),
            "timing": self.timing.to_dict(),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "memory_delta_mb": round(self.memory_delta_mb, 2),
        }
