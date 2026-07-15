"""Tests for registry-backed model matrix execution."""

from __future__ import annotations

import polars as pl
import pytest

from uncertainty_flow.benchmarking.contracts import ReusePolicy, RunRequest, RunStatus
from uncertainty_flow.benchmarking.matrix import ModelMatrixCoordinator


def _request(models):
    return RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=tuple(
            {
                "id": name,
                "provider": name,
                "required": name != "missing-model",
            }
            for name in models
        ),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": "data"},
    )


def test_matrix_publishes_model_specific_results(tmp_path) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i % 5) for i in range(150)], "y": [float(i) for i in range(150)]}
    )

    result = ModelMatrixCoordinator(storage_root=tmp_path).run(
        _request(("conformal-regressor", "quantile-forest")), frame
    )

    assert result.manifest.status == RunStatus.SUCCESS
    assert {item.model_id for item in result.model_results} == {
        "conformal-regressor",
        "quantile-forest",
    }
    assert {item.model_id for item in result.model_results if item.metrics} == {
        "conformal-regressor",
        "quantile-forest",
    }
    assert any(
        "predictions/conformal-regressor.parquet" in path for path in result.manifest.artifacts
    )


def test_matrix_isolates_optional_model_branch_failure(tmp_path) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )

    result = ModelMatrixCoordinator(storage_root=tmp_path).run(
        _request(("conformal-regressor", "missing-model")), frame
    )

    assert result.manifest.status == RunStatus.DEGRADED
    assert result.model_results[-1].status.value == "degraded"
    assert {item.model_id for item in result.model_results if item.metrics} == {
        "conformal-regressor"
    }
    assert result.model_results[-1].model_id == "missing-model"
    assert result.model_results[-1].error == "Unknown model provider 'missing-model'"


def test_matrix_fail_fast_does_not_publish_a_partial_optional_matrix(tmp_path) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    request = _request(("conformal-regressor", "missing-model", "quantile-forest")).model_copy(
        update={"fail_fast": True}
    )

    with pytest.raises(ValueError, match="Unknown model provider"):
        ModelMatrixCoordinator(storage_root=tmp_path).run(request, frame)

    assert not list((tmp_path / "04_platinum" / "runs").glob("*/manifest.json"))


def test_matrix_reuses_a_verified_run(tmp_path) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    coordinator = ModelMatrixCoordinator(storage_root=tmp_path)

    first = coordinator.run(_request(("conformal-regressor", "quantile-forest")), frame)
    second = coordinator.run(_request(("conformal-regressor", "quantile-forest")), frame)

    assert first.reused is False
    assert second.reused is True
    assert second.model_results == first.model_results


def test_matrix_reuse_preserves_optional_failures(tmp_path) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    coordinator = ModelMatrixCoordinator(storage_root=tmp_path)

    first = coordinator.run(_request(("conformal-regressor", "missing-model")), frame)
    second = coordinator.run(_request(("conformal-regressor", "missing-model")), frame)

    assert second.reused is True
    assert second.model_results == first.model_results
    assert second.model_results[-1].status.value == "degraded"


def test_matrix_keeps_variants_of_the_same_provider_distinct(tmp_path) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=(
            {"id": "fast", "provider": "conformal-regressor", "parameters": {}},
            {
                "id": "accurate",
                "provider": "conformal-regressor",
                "parameters": {"n_estimators": 50},
            },
        ),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": str(tmp_path)},
    )

    result = ModelMatrixCoordinator(storage_root=tmp_path).run(request, frame)

    assert [item.model_id for item in result.model_results] == ["fast", "accurate"]
    assert {item.provider for item in result.model_results} == {"conformal-regressor"}
    assert {item.resolved_parameters["n_estimators"] for item in result.model_results} == {
        30,
        50,
    }
    assert any("predictions/fast.parquet" in path for path in result.manifest.artifacts)
    assert any("predictions/accurate.parquet" in path for path in result.manifest.artifacts)


def test_matrix_fail_if_exists_policy_is_enforced(tmp_path) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    coordinator = ModelMatrixCoordinator(storage_root=tmp_path)
    request = _request(("conformal-regressor",))
    coordinator.run(request, frame)

    with pytest.raises(FileExistsError, match="already exists"):
        coordinator.run(
            request.model_copy(update={"reuse_policy": ReusePolicy.FAIL_IF_EXISTS}), frame
        )


def test_matrix_aggregates_rolling_origin_folds(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "timestamp": list(range(300)),
            "feature": [float(i % 5) for i in range(300)],
            "y": [float(i) for i in range(300)],
        }
    )
    request = _request(("conformal-regressor",)).model_copy(
        update={
            "validation": {
                "strategy": "rolling_origin",
                "test_size": 0.1,
                "n_folds": 3,
                "timestamp_column": "timestamp",
            }
        }
    )

    result = ModelMatrixCoordinator(storage_root=tmp_path).run(request, frame)

    assert result.manifest.status == RunStatus.SUCCESS
    assert result.model_results[0].evaluation_row_count == 90
    assert result.model_results[0].model_artifact_ref is None
    assert {item.model_id for item in result.model_results if item.metrics} == {
        "conformal-regressor"
    }


def test_matrix_rejects_empty_model_list_before_execution(tmp_path) -> None:
    frame = pl.DataFrame({"feature": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="At least one model"):
        ModelMatrixCoordinator(storage_root=tmp_path).run_with_lock(_request(()), frame)
