"""Benchmark runner public adapter over deep benchmark flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl

from uncertainty_flow.utils.exceptions import ConfigurationError, DataError

from .configs import BenchmarkConfig
from .datasets import DatasetInfo
from .flow import BenchmarkFlow
from .providers import get_default_providers
from .results import BenchmarkResult, ModelResult
from .sinks import ResultSink

MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    """Decorator to register a class-based benchmark model adapter."""

    def decorator(cls: type) -> type:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


class BenchmarkRunner:
    """Public runner API for executing benchmark flows."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.flow = BenchmarkFlow(
            config=config,
            providers=get_default_providers(),
            class_registry=MODEL_REGISTRY,
        )

        self.df: pl.DataFrame | None = None
        self.ds_info: DatasetInfo | None = None
        self.target: str = ""
        self.results: list[ModelResult] = []
        self.errors: list[dict[str, str]] = []
        self._run_result: BenchmarkResult | None = None

    def load_data(self) -> None:
        loaded = self.flow.load_data()
        self.df = loaded.df
        self.ds_info = loaded.ds_info
        self.target = loaded.target

    def _require_loaded(self):
        if self.flow.loaded is None:
            raise DataError("Data not loaded. Call load_data() first.")

    def run_model(self, model_name: str) -> ModelResult:
        if self.flow.loaded is None:
            self.load_data()
        loaded = self.flow.loaded
        if loaded is None:
            raise DataError("No dataset loaded")
        return self.flow.run_one(model_name, loaded)

    def run_all(
        self,
        model_names: list[str] | None = None,
        allow_partial: bool = False,
    ) -> BenchmarkResult:
        if self.flow.loaded is None:
            self.load_data()

        self._run_result = self.flow.run(model_names=model_names, allow_partial=allow_partial)
        self.results = self._run_result.models
        self.errors = self._run_result.errors
        return self._run_result

    def _sink(self) -> ResultSink:
        return ResultSink(
            result=self._run_result,
            test_size=self.config.test_size,
            auto_tune=self.config.auto_tune,
            target_coverage=self.config.target_coverage,
        )

    def to_dict(self) -> dict[str, Any]:
        return self._sink().to_dict()

    def save_json(self, path: Path | str) -> None:
        self._sink().save_json(path)

    def save_csv(self, path: Path | str) -> None:
        self._sink().save_csv(path)


# Register built-in names in class registry for callers that inspect available names.
class _BuiltinNamePlaceholder:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ConfigurationError("Built-in model placeholders are not directly instantiable.")


for _name in ("quantile-forest", "conformal-regressor", "conformal-forecaster"):
    MODEL_REGISTRY.setdefault(_name, _BuiltinNamePlaceholder)


__all__ = [
    "BenchmarkConfig",
    "ModelResult",
    "BenchmarkResult",
    "BenchmarkRunner",
    "MODEL_REGISTRY",
    "register_model",
]
