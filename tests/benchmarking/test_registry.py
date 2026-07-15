"""Tests for extensible pipeline registries."""

from __future__ import annotations

import polars as pl
import pytest

from uncertainty_flow.benchmarking.registry import (
    DatasetRegistry,
    MetricSpec,
    default_dataset_registry,
    default_metric_registry,
    default_model_registry,
)


def test_default_registries_expose_network_free_adapters(tmp_path) -> None:
    frame = pl.DataFrame({"y": [1, 2, 3]})
    path = tmp_path / "data.parquet"
    frame.write_parquet(path)

    loaded = default_dataset_registry().load("local_parquet", str(path))

    assert loaded.equals(frame)
    assert set(default_dataset_registry().names()) == {"huggingface", "local_parquet"}
    assert set(default_metric_registry().names()) >= {
        "coverage",
        "sharpness",
        "winkler",
        "pinball",
        "crps",
        "mae",
        "rmse",
        "calibration_error",
    }
    assert set(default_model_registry().names()) == {
        "quantile-forest",
        "conformal-regressor",
        "conformal-forecaster",
        "deep-quantile",
        "deep-quantile-torch",
        "transformer-forecaster",
        "bayesian-quantile",
        "linear-regression",
        "ridge-regression",
        "random-forest",
        "gradient-boosting",
        "naive-forecast",
        "moving-average",
    }


def test_model_registry_resolves_defaults_and_rejects_unknown_parameters() -> None:
    registry = default_model_registry()

    resolved = registry.resolve_parameters(
        "conformal-regressor", {"horizon": 2, "n_estimators": 10}
    )

    assert resolved["horizon"] == 2
    assert resolved["n_estimators"] == 10
    assert resolved["random_state"] == 42
    with pytest.raises(ValueError, match="Unknown parameters"):
        registry.resolve_parameters("conformal-regressor", {"unsupported": True})
    for invalid_size in (0.0, 1.0):
        with pytest.raises(ValueError, match="calibration_size"):
            registry.resolve_parameters("conformal-regressor", {"calibration_size": invalid_size})


def test_model_registry_normalizes_distinct_variants_and_rejects_duplicate_ids() -> None:
    registry = default_model_registry()
    variants = registry.resolve_specs(
        (
            {"id": "fast", "provider": "conformal-regressor", "parameters": {}},
            {
                "id": "accurate",
                "provider": "conformal-regressor",
                "parameters": {"n_estimators": 50},
            },
        )
    )

    assert [variant["id"] for variant in variants] == ["fast", "accurate"]
    assert variants[0]["parameters"]["n_estimators"] == 30
    with pytest.raises(ValueError, match="Duplicate model id"):
        registry.resolve_specs(
            (
                {"id": "same", "provider": "conformal-regressor"},
                {"id": "same", "provider": "quantile-forest"},
            )
        )


def test_registry_rejects_unknown_entries() -> None:
    registry = DatasetRegistry()
    registry.register("fixture", lambda _uri: pl.DataFrame({"y": [1]}))

    assert registry.version("fixture") == "v1"
    with pytest.raises(ValueError, match="Unknown dataset provider"):
        registry.load("missing", "unused")


def test_huggingface_adapter_requires_pinned_uri() -> None:
    registry = default_dataset_registry()

    with pytest.raises(ValueError, match="pinned @revision"):
        registry.load("huggingface", "hf://owner/dataset")


def test_metric_spec_can_mark_optional_diagnostics() -> None:
    spec = MetricSpec("feature_leverage", required=False)

    assert spec.required is False
