"""Integration tests for the first verified coordinator run."""

from __future__ import annotations

import json

import polars as pl
import pytest

from uncertainty_flow.benchmarking.contracts import ArtifactType, RunRequest, RunStatus
from uncertainty_flow.benchmarking.coordinator import BenchmarkCoordinator
from uncertainty_flow.benchmarking.storage import LocalArtifactStore


def test_coordinator_publishes_verified_platinum_run(tmp_path) -> None:
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2, "random_seed": 42},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage", "winkler", "pinball"]},
        storage={"provider": "local", "root": str(tmp_path)},
    )
    frame = pl.DataFrame(
        {
            "id": [f"row-{i}" for i in range(150)],
            "feature": [float(i % 5) for i in range(150)],
            "y": [float(i) + 0.25 for i in range(150)],
        }
    )

    result = BenchmarkCoordinator(storage_root=tmp_path).run(request, frame)

    assert result.manifest.status == RunStatus.SUCCESS
    assert result.verification.passed is True
    assert len(result.model_results) == 1
    assert result.model_results[0].provider == "conformal-regressor"
    assert result.model_results[0].evaluation_row_count > 0
    assert result.model_results[0].model_artifact_ref is not None
    assert result.model_results[0].model_artifact_ref.path.endswith("models/conformal-regressor.uf")
    assert (tmp_path / result.model_results[0].model_artifact_ref.path).is_file()
    assert set(result.model_results[0].metrics) == {
        "coverage_90",
        "coverage_80",
        "winkler_90",
        "winkler_80",
        "pinball",
    }
    assert (tmp_path / result.manifest.artifacts[-1]).exists()
    assert (tmp_path / f"04_platinum/runs/{result.manifest.identity.run_id}/metrics.json").exists()


def test_coordinator_reuses_verified_run(tmp_path) -> None:
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": str(tmp_path)},
    )
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    coordinator = BenchmarkCoordinator(storage_root=tmp_path)

    first = coordinator.run(request, frame)
    second = coordinator.run(request, frame)

    assert first.reused is False
    assert second.reused is True
    assert second.manifest.identity.run_id == first.manifest.identity.run_id
    assert second.model_results == first.model_results


def test_coordinator_persists_the_identity_code_version(tmp_path) -> None:
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": str(tmp_path)},
    )
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )

    result = BenchmarkCoordinator(storage_root=tmp_path, code_version="test-build-7").run(
        request, frame
    )
    config_ref = next(
        ref
        for ref in result.manifest.artifact_refs
        if ref.artifact_type == ArtifactType.RESOLVED_CONFIG
    )

    assert json.loads((tmp_path / config_ref.path).read_text())["code_version"] == "test-build-7"


def test_resolved_config_redacts_signing_secret_and_identity_ignores_key_rotation(
    tmp_path,
) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    base = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": str(tmp_path)},
        publication={"manifest_secret": "first-secret"},
    )
    coordinator = BenchmarkCoordinator(storage_root=tmp_path)

    first = coordinator.run(base, frame)
    second = coordinator.run(
        base.model_copy(update={"publication": {"manifest_secret": "rotated-secret"}}),
        frame,
    )
    config_ref = next(
        ref
        for ref in second.manifest.artifact_refs
        if ref.artifact_type == ArtifactType.RESOLVED_CONFIG
    )
    config_text = (tmp_path / config_ref.path).read_text(encoding="utf-8")

    assert first.manifest.identity.run_id == second.manifest.identity.run_id
    assert second.reused is False
    assert "first-secret" not in config_text
    assert "rotated-secret" not in config_text
    assert json.loads(config_text)["request"]["publication"]["manifest_secret"] == "<redacted>"


def test_failed_publication_verification_never_promotes_manifest(tmp_path, monkeypatch) -> None:
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": str(tmp_path)},
    )
    original_write_json = LocalArtifactStore.write_json

    def fail_verification(self, ref, value):
        result = original_write_json(self, ref, value)
        if ref.artifact_type == ArtifactType.VERIFICATION:
            return result.model_copy(update={"verified": False})
        return result

    monkeypatch.setattr(LocalArtifactStore, "write_json", fail_verification)

    with pytest.raises(RuntimeError, match="failed publication verification"):
        BenchmarkCoordinator(storage_root=tmp_path).run(request, frame)
    assert not list((tmp_path / "04_platinum" / "runs").glob("*/manifest.json"))


def test_optional_diagnostic_failure_marks_run_degraded(tmp_path) -> None:
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage"], "diagnostics": {"shap": "optional"}},
        storage={"provider": "local", "root": str(tmp_path)},
    )
    frame = pl.DataFrame(
        {"feature": [float(i) for i in range(150)], "y": [float(i) for i in range(150)]}
    )

    result = BenchmarkCoordinator(storage_root=tmp_path).run(request, frame)

    assert result.manifest.status == RunStatus.DEGRADED
    assert result.manifest.degradation_reasons[0].node == "shap"


def test_coordinator_persists_lineage_and_rebuilds_corrupt_runs(tmp_path) -> None:
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage", "sharpness"], "coverage_levels": [0.95]},
        storage={"provider": "local", "root": str(tmp_path)},
    )
    frame = pl.DataFrame(
        {"feature": list(map(float, range(150))), "y": list(map(float, range(150)))}
    )
    coordinator = BenchmarkCoordinator(storage_root=tmp_path)

    first = coordinator.run(request, frame)

    assert set(first.model_results[0].metrics) == {"coverage_95", "sharpness_95"}
    assert {ref.artifact_type.value for ref in first.manifest.artifact_refs} >= {
        "bronze_dataset",
        "silver_dataset",
        "silver_validation",
        "gold_dataset",
        "gold_splits",
    }
    bronze_ref = next(
        ref for ref in first.manifest.artifact_refs if ref.artifact_type.value == "bronze_dataset"
    )
    silver_ref = next(
        ref for ref in first.manifest.artifact_refs if ref.artifact_type.value == "silver_dataset"
    )
    gold_ref = next(
        ref for ref in first.manifest.artifact_refs if ref.artifact_type.value == "gold_dataset"
    )
    assert "id" not in pl.read_parquet(tmp_path / bronze_ref.path).columns
    assert "id" in pl.read_parquet(tmp_path / silver_ref.path).columns
    assert "_split" in pl.read_parquet(tmp_path / gold_ref.path).columns
    prediction = next(
        ref for ref in first.manifest.artifact_refs if ref.artifact_type.value == "predictions"
    )
    (tmp_path / prediction.path).write_bytes(b"corrupt")

    rebuilt = coordinator.run(request, frame)
    assert rebuilt.reused is False
    assert (tmp_path / prediction.path).read_bytes() != b"corrupt"
