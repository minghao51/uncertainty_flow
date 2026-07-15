"""Tests for versioned Platinum evidence export."""

from __future__ import annotations

import gzip
import json

import polars as pl
import pytest

from uncertainty_flow.benchmarking.contracts import RunRequest
from uncertainty_flow.benchmarking.coordinator import BenchmarkCoordinator
from uncertainty_flow.benchmarking.evidence import EvidenceIndex, export_evidence
from uncertainty_flow.benchmarking.matrix import ModelMatrixCoordinator


def test_export_evidence_writes_indexed_gzip_partition(tmp_path) -> None:
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
    BenchmarkCoordinator(storage_root=tmp_path).run(request, frame)

    output = tmp_path / "evidence"
    index = export_evidence(tmp_path, output)

    assert isinstance(index, EvidenceIndex)
    assert index.partitions[0].records == 1
    partition = output / "runs" / f"{index.partitions[0].key}.jsonl.gz"
    with gzip.open(partition, "rt", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert record["verified"] is True
    assert record["dataset_id"] == "fixture"
    assert json.loads((output / "index.json").read_text())["schema_version"] == 1


def test_export_evidence_emits_one_record_per_matrix_model(tmp_path) -> None:
    request = RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=tuple(
            {"id": name, "provider": name} for name in ("conformal-regressor", "quantile-forest")
        ),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": str(tmp_path)},
    )
    frame = pl.DataFrame(
        {"feature": [float(i % 5) for i in range(150)], "y": [float(i) for i in range(150)]}
    )
    ModelMatrixCoordinator(storage_root=tmp_path).run(request, frame)

    index = export_evidence(tmp_path, tmp_path / "evidence")

    assert index.partitions[0].records == 2
    assert len(index.latest_run_ids) == 1


def test_export_evidence_rejects_corrupt_verified_artifacts(tmp_path) -> None:
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
    result = BenchmarkCoordinator(storage_root=tmp_path).run(request, frame)
    metrics = next(
        ref for ref in result.manifest.artifact_refs if ref.artifact_type.value == "metrics"
    )
    (tmp_path / metrics.path).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt or missing artifacts"):
        export_evidence(tmp_path, tmp_path / "evidence")
