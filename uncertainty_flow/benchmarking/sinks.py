"""Result sink adapters for benchmark flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .results import BenchmarkResult, ModelResult


class ResultSink:
    """Adapter for benchmark result serialization."""

    def __init__(
        self,
        result: BenchmarkResult | None,
        test_size: float,
        auto_tune: bool,
        target_coverage: float,
    ):
        self.result = result
        self.test_size = test_size
        self.auto_tune = auto_tune
        self.target_coverage = target_coverage

    def to_dict(self) -> dict[str, Any]:
        if self.result is None:
            return {"metadata": {}, "results": []}

        results = [self._row(r) for r in self.result.models]
        return {
            "dataset": self.result.dataset_name,
            "metadata": {
                "run_id": self.result.run_id,
                "timestamp": self.result.timestamp,
                "dataset": self.result.dataset_name,
                "domain": self.result.dataset_domain,
                "n_samples": self.result.n_samples,
                "horizon": self.result.horizon,
                "test_size": self.test_size,
                "auto_tune": self.auto_tune,
                "target_coverage": self.target_coverage,
            },
            "errors": self.result.errors,
            "results": results,
        }

    @staticmethod
    def _row(r: ModelResult) -> dict[str, Any]:
        return {
            "model": r.model_name,
            "coverage_90": r.coverage_90,
            "coverage_80": r.coverage_80,
            "sharpness_90": r.sharpness_90,
            "sharpness_80": r.sharpness_80,
            "winkler_90": r.winkler_90,
            "winkler_80": r.winkler_80,
            "pinball_loss": r.pinball_loss,
            "train_time_sec": r.train_time_sec,
            "n_samples": r.n_samples,
            "tuned_params": r.tuned_params,
            "was_tuned": r.was_tuned,
            "validation_coverage_90": r.validation_coverage_90,
            "validation_sharpness_90": r.validation_sharpness_90,
            "validation_winkler_90": r.validation_winkler_90,
            "validation_split_type": r.validation_split_type,
            "validation_strategy": r.validation_strategy,
            "validation_n_splits": r.validation_n_splits,
            "test_split_type": r.test_split_type,
        }

    def save_json(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: Path | str) -> None:
        if self.result is None or not self.result.models:
            return

        rows = []
        for r in self.result.models:
            rows.append(
                {
                    "dataset": self.result.dataset_name,
                    "domain": self.result.dataset_domain,
                    "model": r.model_name,
                    "n_samples": r.n_samples,
                    "horizon": self.result.horizon,
                    "coverage_90": r.coverage_90,
                    "coverage_80": r.coverage_80,
                    "sharpness_90": r.sharpness_90,
                    "sharpness_80": r.sharpness_80,
                    "winkler_90": r.winkler_90,
                    "winkler_80": r.winkler_80,
                    "pinball_loss": r.pinball_loss,
                    "train_time_sec": r.train_time_sec,
                    "was_tuned": r.was_tuned,
                    "tuned_params": str(r.tuned_params),
                    "validation_coverage_90": r.validation_coverage_90,
                    "validation_sharpness_90": r.validation_sharpness_90,
                    "validation_winkler_90": r.validation_winkler_90,
                    "validation_split_type": r.validation_split_type,
                    "validation_strategy": r.validation_strategy,
                    "validation_n_splits": r.validation_n_splits,
                    "test_split_type": r.test_split_type,
                }
            )

        pl.DataFrame(rows).write_csv(path)
