"""Deployment-facing protocols and safe local reference adapters."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from .contracts.runs import RunRequest
from .operations import RunLockManager

logger = logging.getLogger(__name__)


class ObjectStore(Protocol):
    """Minimal object-store contract for artifact backends."""

    def put(self, key: str, source: Path) -> None: ...

    def get(self, key: str, destination: Path) -> None: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...


class LocalObjectStore:
    """Filesystem implementation with object-key path traversal protection."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Object key escapes store root: {key}")
        return self.root / relative

    def put(self, key: str, source: Path) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    def get(self, key: str, destination: Path) -> None:
        source = self._path(key)
        if not source.is_file():
            raise FileNotFoundError(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


@dataclass(frozen=True)
class ScheduleHandle:
    """Opaque scheduler submission identifier."""

    id: str


class Scheduler(Protocol):
    """Scheduler integration boundary."""

    def submit(self, request: RunRequest) -> ScheduleHandle: ...

    def cancel(self, handle: ScheduleHandle) -> None: ...


class RecordingScheduler:
    """Reference scheduler that records requests without executing them."""

    def __init__(self, output: Path | str):
        self.output = Path(output)

    def submit(self, request: RunRequest) -> ScheduleHandle:
        handle = ScheduleHandle(f"schedule-{uuid.uuid4().hex[:12]}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps({"id": handle.id, "request": request.model_dump(mode="json")}) + "\n"
            )
        return handle

    def cancel(self, handle: ScheduleHandle) -> None:
        logger.info("Scheduler cancellation requested", extra={"schedule_id": handle.id})


class AlertSink(Protocol):
    """Alert delivery boundary for degraded and failed runs."""

    def send(self, severity: str, message: str, context: dict[str, str]) -> None: ...


class LoggingAlertSink:
    """Safe default alert sink for local and CI deployments."""

    def send(self, severity: str, message: str, context: dict[str, str]) -> None:
        logger.log(
            logging.ERROR if severity in {"error", "critical"} else logging.WARNING,
            "%s: %s | %s",
            severity,
            message,
            context,
        )


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention policy that never deletes verified published runs by default."""

    retain_verified_days: int | None = None
    delete_unverified: bool = False


def plan_retention(
    root: Path | str,
    policy: RetentionPolicy,
    *,
    now: datetime | None = None,
) -> tuple[Path, ...]:
    """Return deletable run directories without mutating storage."""

    current = now or datetime.now(timezone.utc)
    runs_root = Path(root) / "04_platinum" / "runs"
    candidates: list[Path] = []
    if not runs_root.is_dir():
        return ()
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verified = bool(manifest.get("verification_passed", False))
            finished = datetime.fromisoformat(manifest["finished_at"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            verified = False
            finished = current
        if not verified and policy.delete_unverified:
            candidates.append(run_dir)
        elif (
            verified
            and policy.retain_verified_days is not None
            and finished < current - timedelta(days=policy.retain_verified_days)
        ):
            # Never silently delete verified history; make this explicit for an operator.
            logger.warning("Verified run exceeds retention window: %s", run_dir)
    return tuple(candidates)


def distributed_lock_manager(root: Path | str) -> RunLockManager:
    """Return the default local lock implementation for one deployment root."""

    return RunLockManager(root)
