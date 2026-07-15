"""Immutable pipeline-native execution result contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from .artifacts import ArtifactRef
from .runs import ContractModel, RunManifest
from .verification import RunVerificationReport


class ModelExecutionStatus(StrEnum):
    """Lifecycle status for one requested model branch."""

    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"


class ModelExecutionResult(ContractModel):
    """Typed evidence and outcome for one model/provider execution."""

    model_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    status: ModelExecutionStatus
    required: bool = True
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)
    train_time_sec: float = Field(default=0.0, ge=0.0)
    evaluation_row_count: int = Field(default=0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    degradation_reason: str | None = None
    model_artifact_ref: ArtifactRef | None = None
    prediction_artifact_ref: ArtifactRef | None = None
    metric_artifact_ref: ArtifactRef | None = None


class PipelineRunResult(ContractModel):
    """Shared result shape for single-model and matrix execution."""

    manifest: RunManifest
    verification: RunVerificationReport
    artifacts: tuple[ArtifactRef, ...]
    model_results: tuple[ModelExecutionResult, ...]
    reused: bool = False
