"""Export verified Platinum runs as compact, versioned site evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .contracts import RunManifest, RunVerificationReport
from .contracts.verification import VerificationStatus
from .storage import LocalArtifactStore


class EvidenceModel(BaseModel):
    """Immutable schema boundary for generated site records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RunEvidenceRecord(EvidenceModel):
    """Compact run summary for the evidence portal."""

    schema_version: int = 1
    kind: str = "run"
    run_id: str = Field(min_length=1)
    model_id: str | None = None
    status: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    metrics: dict[str, float]
    verified: bool
    started_at: str = Field(min_length=1)
    lineage: tuple[str, ...] = ()


class EvidencePartition(EvidenceModel):
    """Partition metadata stored in the small index."""

    kind: str
    key: str
    path: str
    records: int = Field(ge=0)
    sha256: str = Field(min_length=1)


class EvidenceIndex(EvidenceModel):
    """Small directly-loadable evidence catalog index."""

    schema_version: int = 1
    generated_at: str
    latest_run_ids: tuple[str, ...]
    partitions: tuple[EvidencePartition, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_records(root: Path, manifest_path: Path) -> list[RunEvidenceRecord]:
    manifest = RunManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not manifest.verification_passed or manifest.status.value not in {"success", "degraded"}:
        return []
    run_root = root / "04_platinum" / "runs" / manifest.identity.run_id
    verification_path = run_root / "verification.json"
    metrics_path = run_root / "metrics.json"
    if not verification_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"Verified run is missing evidence artifacts: {manifest.identity.run_id}")
    verification = RunVerificationReport.model_validate(
        json.loads(verification_path.read_text(encoding="utf-8"))
    )
    if not verification.passed:
        return []
    store = LocalArtifactStore(root)
    if not manifest.artifact_refs:
        raise ValueError(f"Verified run is missing artifact checksums: {manifest.identity.run_id}")
    invalid = [
        ref.path
        for ref in manifest.artifact_refs
        if store.verify(ref).status != VerificationStatus.PASSED
    ]
    if invalid:
        raise ValueError(
            f"Verified run has corrupt or missing artifacts: {manifest.identity.run_id}: {invalid}"
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_results = payload.get("model_results")
    if not isinstance(model_results, list):
        raise ValueError(
            f"Run is missing pipeline-native model results: {manifest.identity.run_id}"
        )
    return [
        RunEvidenceRecord(
            run_id=manifest.identity.run_id,
            model_id=str(result["model_id"]),
            status=str(result.get("status", manifest.status.value)),
            dataset_id=manifest.dataset_id,
            metrics={str(key): float(value) for key, value in result.get("metrics", {}).items()},
            verified=True,
            started_at=manifest.started_at,
            lineage=manifest.artifacts,
        )
        for result in model_results
        if isinstance(result, dict) and "model_id" in result
    ]


def export_evidence(root: Path | str, output: Path | str) -> EvidenceIndex:
    """Export all verified runs into monthly JSONL gzip partitions."""

    root_path = Path(root)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    partition_dir = output_path / "runs"
    partition_dir.mkdir(parents=True, exist_ok=True)
    for stale_partition in partition_dir.glob("*.jsonl.gz"):
        stale_partition.unlink()

    grouped: dict[str, list[RunEvidenceRecord]] = defaultdict(list)
    manifests = sorted((root_path / "04_platinum" / "runs").glob("*/manifest.json"))
    all_records: list[RunEvidenceRecord] = []
    for manifest_path in manifests:
        records = _run_records(root_path, manifest_path)
        for record in records:
            partition_key = datetime.fromisoformat(record.started_at).strftime("%Y-%m")
            grouped[partition_key].append(record)
            all_records.append(record)

    partitions: list[EvidencePartition] = []
    for key, records in sorted(grouped.items()):
        relative_path = Path("runs") / f"{key}.jsonl.gz"
        destination = output_path / relative_path
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
        partitions.append(
            EvidencePartition(
                kind="runs",
                key=key,
                path=f"/evidence/{relative_path.as_posix()}",
                records=len(records),
                sha256=_sha256(destination),
            )
        )

    latest: list[str] = []
    for record in sorted(all_records, key=lambda item: item.started_at, reverse=True):
        if record.run_id not in latest:
            latest.append(record.run_id)
        if len(latest) == 10:
            break
    index = EvidenceIndex(
        generated_at=datetime.now().astimezone().isoformat(),
        latest_run_ids=tuple(latest),
        partitions=tuple(partitions),
    )
    (output_path / "index.json").write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return index
