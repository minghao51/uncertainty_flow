"""Tests for deployment-facing Phase 6 adapters."""

from __future__ import annotations

from pathlib import Path

from uncertainty_flow.benchmarking.contracts import RunRequest
from uncertainty_flow.benchmarking.deployment import (
    LocalObjectStore,
    RecordingScheduler,
    RetentionPolicy,
    plan_retention,
)


def _request() -> RunRequest:
    return RunRequest(
        dataset={"id": "fixture", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": 0.2},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": "data"},
    )


def test_local_object_store_round_trip_and_key_safety(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    store = LocalObjectStore(tmp_path / "objects")

    store.put("runs/run-1/manifest.json", source)
    destination = tmp_path / "out.txt"
    store.get("runs/run-1/manifest.json", destination)

    assert destination.read_text(encoding="utf-8") == "payload"
    assert store.exists("runs/run-1/manifest.json")

    try:
        store.exists("../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal key was accepted")


def test_recording_scheduler_is_replayable(tmp_path) -> None:
    scheduler = RecordingScheduler(tmp_path / "schedules.jsonl")

    handle = scheduler.submit(_request())
    scheduler.cancel(handle)

    assert handle.id.startswith("schedule-")
    assert (tmp_path / "schedules.jsonl").read_text(encoding="utf-8").count(handle.id) == 1


def test_retention_only_plans_unverified_deletion(tmp_path) -> None:
    failed = tmp_path / "04_platinum" / "runs" / "failed"
    failed.mkdir(parents=True)
    (failed / "manifest.json").write_text('{"verification_passed": false}', encoding="utf-8")

    candidates = plan_retention(
        tmp_path,
        RetentionPolicy(retain_verified_days=1, delete_unverified=True),
    )

    assert candidates == (Path(failed),)
