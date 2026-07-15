"""Tests for Phase 1 benchmark contracts and stable identities."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from uncertainty_flow.benchmarking.contracts import (
    DatasetRef,
    LeakageCheckResult,
    ModelExecutionResult,
    ModelExecutionStatus,
    SplitAssignment,
    SplitStrategy,
    ValidationPlan,
)
from uncertainty_flow.benchmarking.identity import canonical_json, content_hash, derive_identity


def test_contracts_are_immutable_and_round_trip() -> None:
    dataset = DatasetRef(
        dataset_id="example",
        provider="local_parquet",
        source_uri="data/input/example.parquet",
    )

    assert DatasetRef.model_validate_json(dataset.model_dump_json()) == dataset
    with pytest.raises(Exception):
        dataset.dataset_id = "changed"  # type: ignore[misc]


def test_model_execution_result_is_immutable_and_round_trips() -> None:
    result = ModelExecutionResult(
        model_id="model-1",
        provider="provider-1",
        status=ModelExecutionStatus.SUCCESS,
        resolved_parameters={"seed": 42},
        train_time_sec=0.25,
        evaluation_row_count=10,
        metrics={"coverage_90": 0.9},
    )

    assert ModelExecutionResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(Exception):
        result.status = ModelExecutionStatus.FAILED  # type: ignore[misc]


def test_validation_plan_rejects_duplicate_observation_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ValidationPlan(
            validation_plan_id="plan-1",
            strategy=SplitStrategy.RANDOM_HOLDOUT,
            assignments=(
                SplitAssignment(observation_id="row-1", split="train"),
                SplitAssignment(observation_id="row-1", split="test"),
            ),
            leakage_check=LeakageCheckResult(passed=False, checked_rows=2),
        )


def test_canonical_json_and_hash_are_order_independent() -> None:
    left = {"b": [2, 1], "a": {"z": True, "y": None}}
    right = {"a": {"y": None, "z": True}, "b": [2, 1]}

    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)


def test_canonical_json_supports_temporal_and_non_finite_dataset_values() -> None:
    value = {
        "timestamp": datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
        "values": [np.int64(1), float("nan"), float("inf")],
    }

    payload = canonical_json(value)

    assert "2026-07-14T12:00:00+00:00" in payload
    assert payload.count("__non_finite_float__") == 2
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_identity_changes_for_semantic_inputs_but_not_timestamps() -> None:
    kwargs = {
        "source_checksum": "source-1",
        "ingestion_contract_version": "bronze-v1",
        "validation_contract": {"target": "y"},
        "transformation_version": "transform-v1",
        "split_configuration": {"strategy": "random_holdout", "seed": 42},
        "model_specification": [{"id": "model-1"}],
        "evaluation_specification": {"metrics": ["coverage"]},
        "code_version": "code-1",
        "dataset_specification": {"provider": "local_parquet", "version": "v1"},
    }

    first = derive_identity(**kwargs)
    second = derive_identity(**kwargs)
    changed = derive_identity(**{**kwargs, "model_specification": [{"id": "model-2"}]})

    assert first == second
    assert first["dataset_version"] == changed["dataset_version"]
    assert first["run_id"] != changed["run_id"]
