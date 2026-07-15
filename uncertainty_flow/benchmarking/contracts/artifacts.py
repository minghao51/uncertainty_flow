"""Artifact references and materialization contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Base for immutable artifact metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ArtifactType(StrEnum):
    """Artifact categories used by the medallion layout."""

    BRONZE_DATASET = "bronze_dataset"
    SILVER_DATASET = "silver_dataset"
    SILVER_VALIDATION = "silver_validation"
    GOLD_DATASET = "gold_dataset"
    GOLD_SPLITS = "gold_splits"
    MODEL = "model"
    PREDICTIONS = "predictions"
    METRICS = "metrics"
    CALIBRATION = "calibration"
    DIAGNOSTICS = "diagnostics"
    RESOLVED_CONFIG = "resolved_config"
    MANIFEST = "manifest"
    VERIFICATION = "verification"


class ArtifactChecksum(ContractModel):
    """Checksum and byte size for a materialized artifact."""

    algorithm: str = "sha256"
    digest: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class ArtifactRef(ContractModel):
    """Logical artifact reference resolved by an artifact store."""

    artifact_type: ArtifactType
    path: str = Field(min_length=1)
    checksum: ArtifactChecksum | None = None
    schema_version: str = Field(min_length=1)


class MaterializationResult(ContractModel):
    """Result of an atomic artifact write."""

    ref: ArtifactRef
    created: bool
    verified: bool
