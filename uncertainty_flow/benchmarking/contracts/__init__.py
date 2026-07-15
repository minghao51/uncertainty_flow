"""Immutable metadata contracts for the benchmarking pipeline."""

from .artifacts import ArtifactChecksum, ArtifactRef, ArtifactType, MaterializationResult
from .datasets import ColumnRole, DatasetManifest, DatasetRef, DatasetSchema, ValidationIssue
from .results import ModelExecutionResult, ModelExecutionStatus, PipelineRunResult
from .runs import (
    DegradationReason,
    ResolvedRunConfig,
    ReusePolicy,
    RunIdentity,
    RunManifest,
    RunRequest,
    RunStatus,
)
from .validation import LeakageCheckResult, SplitAssignment, SplitStrategy, ValidationPlan
from .verification import (
    RunVerificationReport,
    VerificationCheck,
    VerificationSeverity,
    VerificationStatus,
)

__all__ = [
    "ArtifactChecksum",
    "ArtifactRef",
    "ArtifactType",
    "ColumnRole",
    "DatasetManifest",
    "DatasetRef",
    "DatasetSchema",
    "DegradationReason",
    "LeakageCheckResult",
    "MaterializationResult",
    "ModelExecutionResult",
    "ModelExecutionStatus",
    "ResolvedRunConfig",
    "RunIdentity",
    "RunManifest",
    "RunRequest",
    "RunStatus",
    "ReusePolicy",
    "PipelineRunResult",
    "SplitAssignment",
    "SplitStrategy",
    "ValidationIssue",
    "ValidationPlan",
    "VerificationCheck",
    "VerificationSeverity",
    "VerificationStatus",
    "RunVerificationReport",
]
