"""Tests for Phase 6 local operational controls."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest

from uncertainty_flow.benchmarking.contracts import (
    RunIdentity,
    RunManifest,
    RunStatus,
)
from uncertainty_flow.benchmarking.operations import (
    NodeEvent,
    NodeEventWriter,
    RunLockManager,
    manifest_authenticity_valid,
    prune_unverified_runs,
    sign_manifest,
    verify_manifest_signature,
)


def _manifest(run_id: str = "run-1") -> RunManifest:
    return RunManifest(
        identity=RunIdentity(
            dataset_version="dataset",
            silver_version="silver",
            validation_plan_id="plan",
            run_id=run_id,
        ),
        dataset_id="fixture",
        dataset_domain="synthetic",
        status=RunStatus.SUCCESS,
        started_at=datetime.now(timezone.utc).isoformat(),
        resolved_config_hash="config",
        verification_passed=True,
    )


def test_run_lock_rejects_active_and_recovers_stale_lock(tmp_path) -> None:
    manager = RunLockManager(tmp_path, stale_after_seconds=1)
    with manager.lock("run-1"):
        with pytest.raises(RuntimeError, match="already locked"):
            with manager.lock("run-1"):
                pass
    stale = tmp_path / "_locks" / "run-2.lock"
    stale.write_text(
        json.dumps(
            {
                "token": "old",
                "created_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    with manager.lock("run-2"):
        assert stale.exists()


def test_node_events_are_readable_as_compressed_jsonl(tmp_path) -> None:
    path = tmp_path / "events.jsonl.gz"
    NodeEventWriter(path).write(
        NodeEvent(run_id="run-1", node="fit", status="success", timestamp="now")
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline())["node"] == "fit"


def test_manifest_signature_and_conservative_gc(tmp_path) -> None:
    manifest = _manifest()
    signature = sign_manifest(manifest, b"secret")
    assert verify_manifest_signature(manifest, signature, b"secret")
    assert not verify_manifest_signature(manifest, signature, b"wrong")

    failed_dir = tmp_path / "04_platinum" / "runs" / "failed-run"
    failed_dir.mkdir(parents=True)
    (failed_dir / "manifest.json").write_text(
        manifest.model_copy(
            update={"status": RunStatus.FAILED, "verification_passed": False}
        ).model_dump_json(),
        encoding="utf-8",
    )
    assert prune_unverified_runs(tmp_path, dry_run=True) == ("failed-run",)
    prune_unverified_runs(tmp_path, dry_run=False)
    assert not failed_dir.exists()


def test_gc_removes_abandoned_staging_but_preserves_active_runs(tmp_path) -> None:
    abandoned = tmp_path / ".staging" / "abandoned"
    abandoned.mkdir(parents=True)
    manager = RunLockManager(tmp_path)

    with manager.lock("active"):
        active = tmp_path / ".staging" / "active"
        active.mkdir(parents=True)
        assert prune_unverified_runs(tmp_path, dry_run=True) == ("abandoned",)
        prune_unverified_runs(tmp_path, dry_run=False)
        assert active.is_dir()

    assert not abandoned.exists()


def test_manifest_signature_excludes_embedded_signature() -> None:
    manifest = _manifest()
    signature = sign_manifest(manifest, b"secret")
    signed = manifest.model_copy(update={"manifest_signature": signature})

    assert verify_manifest_signature(signed, signature, b"secret")
    assert manifest_authenticity_valid(signed, b"secret")
    assert not manifest_authenticity_valid(signed, None)
    assert not manifest_authenticity_valid(manifest, b"secret")
    assert manifest_authenticity_valid(manifest, None)
