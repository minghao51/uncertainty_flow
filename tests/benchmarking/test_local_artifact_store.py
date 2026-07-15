"""Tests for atomic local artifact persistence and verification."""

from __future__ import annotations

import polars as pl

from uncertainty_flow.benchmarking.contracts import ArtifactRef, ArtifactType, VerificationStatus
from uncertainty_flow.benchmarking.storage import LocalArtifactStore


def test_local_store_writes_and_verifies_json(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = ArtifactRef(
        artifact_type=ArtifactType.MANIFEST,
        path="04_platinum/runs/run-1/manifest.json",
        schema_version="v1",
    )

    result = store.write_json(ref, {"run_id": "run-1", "status": "success"})

    assert result.created is True
    assert result.verified is True
    assert store.read_json(result.ref)["run_id"] == "run-1"
    assert store.verify(result.ref).status == VerificationStatus.PASSED


def test_local_store_writes_binary_artifacts(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = ArtifactRef(artifact_type=ArtifactType.MODEL, path="model.uf", schema_version="v1")

    result = store.write_bytes(ref, b"model-bytes")

    assert (tmp_path / "model.uf").read_bytes() == b"model-bytes"
    assert store.verify(result.ref).status == VerificationStatus.PASSED


def test_local_store_detects_checksum_corruption(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = ArtifactRef(
        artifact_type=ArtifactType.METRICS,
        path="04_platinum/runs/run-1/metrics.json",
        schema_version="v1",
    )
    result = store.write_json(ref, {"coverage": 0.9})
    (tmp_path / ref.path).write_text('{"coverage": 0.1}', encoding="utf-8")

    check = store.verify(result.ref)

    assert check.status == VerificationStatus.FAILED
    assert check.check_id == "artifact.checksum"


def test_local_store_writes_parquet_atomically(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = ArtifactRef(
        artifact_type=ArtifactType.GOLD_DATASET,
        path="03_gold/example/version-1/observations.parquet",
        schema_version="v1",
    )
    frame = pl.DataFrame({"id": ["a", "b"], "y": [1.0, 2.0]})

    result = store.write_table(ref, frame)

    assert result.verified is True
    assert store.read_table(result.ref).equals(frame)


def test_staged_store_promotes_manifest_last(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path)
    staged = store.staging("run-1")
    artifact = ArtifactRef(
        artifact_type=ArtifactType.METRICS,
        path="04_platinum/runs/run-1/metrics.json",
        schema_version="v1",
    )
    manifest = ArtifactRef(
        artifact_type=ArtifactType.MANIFEST,
        path="04_platinum/runs/run-1/manifest.json",
        schema_version="v1",
    )
    staged.write_json(artifact, {"coverage": 0.9})
    staged.write_json(manifest, {"status": "success"})

    store.promote(staged, "run-1")

    assert store.exists(artifact)
    assert store.exists(manifest)
    assert not staged.root.exists()
