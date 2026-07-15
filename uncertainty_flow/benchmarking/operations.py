"""Local operational hardening primitives for benchmark runs."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import secrets
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .contracts.runs import RunManifest
from .identity import canonical_json


@dataclass(frozen=True)
class LockHandle:
    """Owned run-lock handle."""

    path: Path
    token: str


class RunLockManager:
    """Acquire run-scoped local locks with stale-lock recovery."""

    def __init__(self, root: Path | str, stale_after_seconds: float = 3600.0):
        self.root = Path(root) / "_locks"
        self.stale_after_seconds = stale_after_seconds

    @contextmanager
    def lock(self, run_id: str) -> Iterator[LockHandle]:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{run_id}.lock"
        token = secrets.token_hex(16)
        payload = {
            "token": token,
            "run_id": run_id,
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._create_or_recover(path, payload)
        except FileExistsError as error:
            raise RuntimeError(f"Run {run_id} is already locked") from error
        handle = LockHandle(path=path, token=token)
        try:
            yield handle
        finally:
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                current = {}
            if current.get("token") == token:
                path.unlink(missing_ok=True)

    def _create_or_recover(self, path: Path, payload: dict[str, object]) -> None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not self._is_stale(path):
                raise
            path.unlink(missing_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())

    def _is_stale(self, path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(str(payload["created_at"]))
            return time.time() - created.timestamp() > self.stale_after_seconds
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return True

    def is_active(self, run_id: str) -> bool:
        """Return whether a non-stale lock currently protects a run."""

        path = self.root / f"{run_id}.lock"
        return path.is_file() and not self._is_stale(path)


@dataclass(frozen=True)
class NodeEvent:
    """Structured execution event."""

    run_id: str
    node: str
    status: str
    timestamp: str
    duration_ms: float | None = None
    message: str | None = None


class NodeEventWriter:
    """Append structured node events to a compressed JSONL stream."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: NodeEvent) -> None:
        with gzip.open(self.path, "at", encoding="utf-8") as handle:
            handle.write(json.dumps(event.__dict__, sort_keys=True) + "\n")


def sign_manifest(manifest: RunManifest, secret: bytes) -> str:
    """Create an HMAC-SHA256 signature for a canonical run manifest."""

    payload = manifest.model_dump(mode="json", exclude={"manifest_signature"})
    return hmac.new(secret, canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def verify_manifest_signature(manifest: RunManifest, signature: str, secret: bytes) -> bool:
    """Verify a manifest signature using constant-time comparison."""

    expected = sign_manifest(manifest, secret)
    return hmac.compare_digest(expected, signature)


def manifest_authenticity_valid(manifest: RunManifest, secret: bytes | None) -> bool:
    """Require signed manifests and signing-enabled requests to authenticate symmetrically."""

    if manifest.manifest_signature is None:
        return secret is None
    return secret is not None and verify_manifest_signature(
        manifest, manifest.manifest_signature, secret
    )


def publication_secret(publication: dict[str, object]) -> bytes | None:
    """Resolve an optional per-request manifest-signing secret."""

    value = publication.get("manifest_secret")
    if isinstance(value, str) and value:
        return value.encode("utf-8")
    return None


def prune_unverified_runs(root: Path | str, *, dry_run: bool = True) -> tuple[str, ...]:
    """Remove only failed/incomplete run directories; never delete published runs."""

    root_path = Path(root)
    removed: set[str] = set()
    runs_root = root_path / "04_platinum" / "runs"
    if runs_root.is_dir():
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                should_remove = True
            else:
                try:
                    manifest = RunManifest.model_validate(
                        json.loads(manifest_path.read_text(encoding="utf-8"))
                    )
                    should_remove = (
                        not manifest.verification_passed or manifest.status.value == "failed"
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    should_remove = True
            if should_remove:
                removed.add(run_dir.name)
                if not dry_run:
                    shutil.rmtree(run_dir)

    staging_root = root_path / ".staging"
    lock_manager = RunLockManager(root_path)
    if staging_root.is_dir():
        for run_dir in sorted(path for path in staging_root.iterdir() if path.is_dir()):
            if lock_manager.is_active(run_dir.name):
                continue
            removed.add(run_dir.name)
            if not dry_run:
                shutil.rmtree(run_dir)
    return tuple(sorted(removed))
