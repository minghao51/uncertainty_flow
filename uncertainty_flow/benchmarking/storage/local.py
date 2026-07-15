"""Local filesystem artifact store with atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from ..contracts.artifacts import ArtifactChecksum, ArtifactRef, MaterializationResult
from ..contracts.verification import (
    VerificationCheck,
    VerificationSeverity,
    VerificationStatus,
)


class LocalArtifactStore:
    """Persist JSON and Parquet artifacts under a local root."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, ref: ArtifactRef) -> Path:
        relative_path = Path(ref.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Artifact path escapes store root: {ref.path}")
        return self.root / relative_path

    def exists(self, ref: ArtifactRef) -> bool:
        return self._path(ref).is_file()

    def staging(self, run_id: str) -> "LocalArtifactStore":
        """Return an isolated store for a run that is not reusable yet."""

        return LocalArtifactStore(self.root / ".staging" / run_id)

    def clear_staging(self, run_id: str) -> None:
        """Discard an incomplete prior attempt before rebuilding the same run."""

        relative = Path(run_id)
        if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
            raise ValueError(f"Invalid staging run id: {run_id}")
        shutil.rmtree(self.root / ".staging" / run_id, ignore_errors=True)

    def promote(self, staged: "LocalArtifactStore", run_id: str) -> None:
        """Promote staged files, moving the final manifest last.

        A partial promotion can leave immutable upstream artifacts behind, but it
        cannot create a reusable run because the final manifest is the last move.
        """

        if staged.root == self.root or not staged.root.is_dir():
            raise ValueError("Staged store must be a separate existing directory")
        manifest_relative = Path("04_platinum") / "runs" / run_id / "manifest.json"
        files = sorted(path for path in staged.root.rglob("*") if path.is_file())
        manifest = staged.root / manifest_relative
        if not manifest.is_file():
            raise ValueError("Staged run has no final manifest")
        for path in files:
            relative = path.relative_to(staged.root)
            if relative == manifest_relative:
                continue
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination)
        destination = self.root / manifest_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(manifest, destination)
        shutil.rmtree(staged.root, ignore_errors=True)

    def read_json(self, ref: ArtifactRef) -> Mapping[str, Any]:
        with self._path(ref).open(encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, ref: ArtifactRef, value: Mapping[str, Any]) -> MaterializationResult:
        payload = json.dumps(value, sort_keys=True, indent=2).encode("utf-8")
        return self._write_bytes(ref, payload)

    def write_bytes(self, ref: ArtifactRef, value: bytes) -> MaterializationResult:
        """Write an arbitrary binary artifact through the checksummed path."""

        return self._write_bytes(ref, value)

    def read_table(self, ref: ArtifactRef) -> pl.DataFrame:
        return pl.read_parquet(self._path(ref))

    def write_table(self, ref: ArtifactRef, value: pl.DataFrame) -> MaterializationResult:
        destination = self._path(ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            value.write_parquet(temporary)
            result = self._publish(ref, temporary)
        finally:
            temporary.unlink(missing_ok=True)
        return result

    def verify(self, ref: ArtifactRef) -> VerificationCheck:
        path = self._path(ref)
        if not path.is_file():
            return VerificationCheck(
                check_id="artifact.exists",
                status=VerificationStatus.FAILED,
                severity=VerificationSeverity.ERROR,
                target=ref.path,
                failure_message="Artifact file is missing",
            )
        if ref.checksum is not None:
            actual = self._checksum(path)
            if actual.digest != ref.checksum.digest:
                return VerificationCheck(
                    check_id="artifact.checksum",
                    status=VerificationStatus.FAILED,
                    severity=VerificationSeverity.ERROR,
                    target=ref.path,
                    evidence={"expected": ref.checksum.digest, "actual": actual.digest},
                    failure_message="Artifact checksum does not match manifest",
                )
        return VerificationCheck(
            check_id="artifact.integrity",
            status=VerificationStatus.PASSED,
            severity=VerificationSeverity.INFO,
            target=ref.path,
        )

    def _write_bytes(self, ref: ArtifactRef, payload: bytes) -> MaterializationResult:
        destination = self._path(ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return self._publish(ref, temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def _publish(self, ref: ArtifactRef, temporary: Path) -> MaterializationResult:
        destination = self._path(ref)
        checksum = self._checksum(temporary)
        os.replace(temporary, destination)
        published_ref = ref.model_copy(update={"checksum": checksum})
        verified = self.verify(published_ref).status == VerificationStatus.PASSED
        return MaterializationResult(ref=published_ref, created=True, verified=verified)

    @staticmethod
    def _checksum(path: Path) -> ArtifactChecksum:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return ArtifactChecksum(digest=digest.hexdigest(), size_bytes=path.stat().st_size)
