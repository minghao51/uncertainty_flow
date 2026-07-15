"""Extensible dataset, model, and metric registries for the pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, cast

import polars as pl

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.metrics import score

from .providers import BenchmarkModelProvider, get_default_providers

DatasetLoader = Callable[[str], pl.DataFrame]
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _load_huggingface(uri: str) -> pl.DataFrame:
    """Load a pinned HuggingFace reference using the existing dataset adapter."""

    if not uri.startswith("hf://"):
        raise ValueError("HuggingFace dataset URIs must use hf://<dataset>@<revision>")
    reference = uri.removeprefix("hf://")
    name, separator, revision = reference.rpartition("@")
    if not separator or not name or not revision:
        raise ValueError("HuggingFace dataset URIs must include a pinned @revision")
    from .datasets import load_dataset

    frame, _ = load_dataset(name, revision=revision)
    return frame


@dataclass(frozen=True)
class MetricSpec:
    """Metric registration metadata."""

    name: str
    required: bool = True
    supports_multivariate: bool = False
    per_level: bool = False
    evaluator: Callable[[DistributionPrediction, pl.Series, float], float] | None = None


@dataclass(frozen=True)
class ParameterSpec:
    """Typed and bounded parameter declaration for one provider."""

    name: str
    value_type: type
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    minimum_exclusive: bool = False
    maximum_exclusive: bool = False


def _validate_parameter(spec: ParameterSpec, value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, spec.value_type):
        raise ValueError(f"Parameter {spec.name!r} must be {spec.value_type.__name__}")
    numeric_value = float(value) if isinstance(value, int | float) else None
    if spec.minimum is not None and (numeric_value is None or numeric_value < spec.minimum):
        raise ValueError(f"Parameter {spec.name!r} must be >= {spec.minimum}")
    if spec.minimum is not None and spec.minimum_exclusive and numeric_value == spec.minimum:
        raise ValueError(f"Parameter {spec.name!r} must be > {spec.minimum}")
    if spec.maximum is not None and (numeric_value is None or numeric_value > spec.maximum):
        raise ValueError(f"Parameter {spec.name!r} must be <= {spec.maximum}")
    if spec.maximum is not None and spec.maximum_exclusive and numeric_value == spec.maximum:
        raise ValueError(f"Parameter {spec.name!r} must be < {spec.maximum}")
    return value


class ModelProviderRegistry:
    """Registry that decouples model discovery from the coordinator."""

    def __init__(
        self,
        providers: dict[str, BenchmarkModelProvider] | None = None,
        parameter_schemas: dict[str, tuple[ParameterSpec, ...]] | None = None,
    ):
        self._providers = dict(providers or {})
        self._parameter_schemas = dict(parameter_schemas or {})

    def register(
        self,
        provider: BenchmarkModelProvider,
        parameters: tuple[ParameterSpec, ...] = (),
    ) -> None:
        self._providers[provider.name] = provider
        self._parameter_schemas[provider.name] = parameters

    def get(self, name: str) -> BenchmarkModelProvider:
        try:
            return self._providers[name]
        except KeyError as error:
            raise ValueError(f"Unknown model provider {name!r}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def resolve_parameters(self, name: str, parameters: dict[str, object]) -> dict[str, object]:
        """Validate and fill defaults for a registered provider."""

        self.get(name)
        schema = self._parameter_schemas.get(name, ())
        known = {item.name for item in schema}
        unknown = sorted(set(parameters) - known)
        if unknown:
            raise ValueError(f"Unknown parameters for {name!r}: {', '.join(unknown)}")
        resolved = {item.name: item.default for item in schema}
        resolved.update(parameters)
        return {
            item.name: _validate_parameter(item, resolved[item.name])
            for item in schema
            if item.name in resolved
        }

    def resolve_specs(self, model_specs: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        """Normalize model identities and parameters before hashing or execution."""

        resolved_specs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw_spec in model_specs:
            provider = raw_spec.get("provider", raw_spec.get("id"))
            if not isinstance(provider, str) or not provider:
                raise ValueError("Every model requires a non-empty provider")
            model_id = raw_spec.get("id", provider)
            if not isinstance(model_id, str) or not _MODEL_ID_PATTERN.fullmatch(model_id):
                raise ValueError(
                    "Model IDs must start with an alphanumeric character and contain only "
                    "letters, numbers, '.', '_', or '-'"
                )
            if model_id in seen_ids:
                raise ValueError(f"Duplicate model id {model_id!r}")
            seen_ids.add(model_id)
            required = raw_spec.get("required", True)
            if not isinstance(required, bool):
                raise ValueError(f"Model {model_id!r} required must be a boolean")
            parameters = raw_spec.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError(f"Parameters for {model_id!r} must be an object")
            if provider in self._providers:
                parameters = self.resolve_parameters(provider, parameters)
            elif required:
                self.get(provider)
            resolved_specs.append(
                {
                    **raw_spec,
                    "id": model_id,
                    "provider": provider,
                    "required": required,
                    "parameters": parameters,
                }
            )
        return tuple(resolved_specs)


class DatasetRegistry:
    """Registry for explicit dataset loaders."""

    def __init__(self):
        self._loaders: dict[str, tuple[DatasetLoader, str]] = {}

    def register(self, provider: str, loader: DatasetLoader, version: str = "v1") -> None:
        self._loaders[provider] = (loader, version)

    def load(self, provider: str, uri: str) -> pl.DataFrame:
        try:
            loader, _ = self._loaders[provider]
            return loader(uri)
        except KeyError as error:
            raise ValueError(f"Unknown dataset provider {provider!r}") from error

    def version(self, provider: str) -> str:
        try:
            return self._loaders[provider][1]
        except KeyError as error:
            raise ValueError(f"Unknown dataset provider {provider!r}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaders))


class MetricRegistry:
    """Registry of required and optional evaluation metrics."""

    def __init__(self, specs: tuple[MetricSpec, ...] = ()):
        self._specs = {spec.name: spec for spec in specs}

    def register(self, spec: MetricSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> MetricSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise ValueError(f"Unknown metric {name!r}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def evaluate(
        self,
        name: str,
        prediction: DistributionPrediction,
        y_true: pl.Series,
        confidence: float = 0.9,
    ) -> float:
        spec = self.get(name)
        if spec.evaluator is None:
            raise ValueError(f"Metric {name!r} has no evaluator")
        return float(spec.evaluator(prediction, y_true, confidence))


def _score_metric(name: str) -> Callable[[DistributionPrediction, pl.Series, float], float]:
    return lambda prediction, y_true, confidence: float(
        cast(float, score(prediction, y_true, name, confidence=confidence))
    )


def _sharpness_metric(
    prediction: DistributionPrediction, _y_true: pl.Series, confidence: float
) -> float:
    lower, upper = prediction.interval_bounds(confidence)
    return float(cast(float, (upper - lower).mean()))


def default_model_registry() -> ModelProviderRegistry:
    """Return built-in model providers without exposing a mutable global."""

    common = (ParameterSpec("random_state", int, 42),)
    calibrated = common + (
        ParameterSpec(
            "calibration_size",
            float,
            0.2,
            minimum=0.0,
            maximum=1.0,
            minimum_exclusive=True,
            maximum_exclusive=True,
        ),
    )
    ensemble = calibrated + (ParameterSpec("n_estimators", int, 30, minimum=1),)
    providers = get_default_providers()
    schemas = {
        "quantile-forest": common
        + (
            ParameterSpec("horizon", int, 3, minimum=1),
            ParameterSpec("n_estimators", int, 30, minimum=1),
        ),
        "conformal-regressor": ensemble + (ParameterSpec("horizon", int, 3, minimum=1),),
        "conformal-forecaster": ensemble
        + (
            ParameterSpec("horizon", int, 3, minimum=1),
            ParameterSpec("lags", int, 2, minimum=1),
        ),
        "linear-regression": calibrated,
        "ridge-regression": calibrated + (ParameterSpec("alpha", float, 1.0, minimum=0.0),),
        "random-forest": ensemble,
        "gradient-boosting": ensemble,
        "naive-forecast": common + (ParameterSpec("horizon", int, 3, minimum=1),),
        "moving-average": common
        + (
            ParameterSpec("horizon", int, 3, minimum=1),
            ParameterSpec("window", int, 5, minimum=1),
        ),
        "deep-quantile": common + (ParameterSpec("trunk_max_iter", int, 300, minimum=1),),
        "deep-quantile-torch": common
        + (
            ParameterSpec("epochs", int, 100, minimum=1),
            ParameterSpec("learning_rate", float, 0.001, minimum=0.0, minimum_exclusive=True),
        ),
        "transformer-forecaster": calibrated + (ParameterSpec("horizon", int, 3, minimum=1),),
        "bayesian-quantile": common
        + (
            ParameterSpec("n_warmup", int, 1000, minimum=1),
            ParameterSpec("n_samples", int, 2000, minimum=1),
            ParameterSpec("prior_width", float, 10.0, minimum=0.0, minimum_exclusive=True),
        ),
    }
    return ModelProviderRegistry(providers, schemas)


def default_metric_registry() -> MetricRegistry:
    """Return the initial required metric set."""

    return MetricRegistry(
        (
            MetricSpec("coverage", per_level=True, evaluator=_score_metric("coverage")),
            MetricSpec("sharpness", per_level=True, evaluator=_sharpness_metric),
            MetricSpec("winkler", per_level=True, evaluator=_score_metric("winkler")),
            MetricSpec("pinball", evaluator=_score_metric("pinball")),
            MetricSpec("crps", evaluator=_score_metric("crps")),
            MetricSpec("mae", evaluator=_score_metric("mae")),
            MetricSpec("rmse", evaluator=_score_metric("rmse")),
            MetricSpec(
                "calibration_error",
                per_level=True,
                evaluator=_score_metric("calibration_error"),
            ),
        )
    )


def default_dataset_registry() -> DatasetRegistry:
    """Return local and explicitly pinned remote dataset adapters."""

    registry = DatasetRegistry()
    registry.register("local_parquet", pl.read_parquet)
    registry.register("huggingface", _load_huggingface, version="huggingface-v1")
    return registry
