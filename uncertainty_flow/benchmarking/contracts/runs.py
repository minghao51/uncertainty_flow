"""Run request, identity, and lifecycle contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import ArtifactRef


class ContractModel(BaseModel):
    """Base for immutable run metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RunStatus(StrEnum):
    """Allowed run lifecycle states."""

    PLANNED = "planned"
    RUNNING = "running"
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReusePolicy(StrEnum):
    """Behavior when the content-derived run already exists."""

    REUSE_VERIFIED = "reuse_verified"
    FAIL_IF_EXISTS = "fail_if_exists"
    RERUN = "rerun"


class DegradationReason(ContractModel):
    """Recorded failure of an optional branch."""

    node: str = Field(min_length=1)
    exception_category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    evidence_impact: str = Field(min_length=1)
    remediation: str = Field(min_length=1)


class RunRequest(ContractModel):
    """User-level request before defaults and identities are resolved."""

    mode: str = "benchmark"
    dataset: dict[str, Any]
    validation: dict[str, Any]
    models: tuple[dict[str, Any], ...]
    evaluation: dict[str, Any]
    storage: dict[str, Any]
    publication: dict[str, Any] = Field(default_factory=dict)
    reuse_policy: ReusePolicy = ReusePolicy.REUSE_VERIFIED
    fail_fast: bool = False


class ResolvedRunConfig(ContractModel):
    """Canonical configuration supplied to DAG construction and nodes."""

    request: RunRequest
    config_hash: str = Field(min_length=1)
    code_version: str = Field(min_length=1)


class RunIdentity(ContractModel):
    """Content-derived identities for a run and its upstream inputs."""

    dataset_version: str = Field(min_length=1)
    silver_version: str = Field(min_length=1)
    validation_plan_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


class RunManifest(ContractModel):
    """Final or in-progress machine-readable run manifest."""

    identity: RunIdentity
    dataset_id: str = Field(min_length=1)
    dataset_domain: str = Field(min_length=1)
    status: RunStatus
    started_at: str = Field(min_length=1)
    finished_at: str | None = None
    resolved_config_hash: str = Field(min_length=1)
    artifacts: tuple[str, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()
    degradation_reasons: tuple[DegradationReason, ...] = ()
    verification_passed: bool = False
    manifest_signature: str | None = None
