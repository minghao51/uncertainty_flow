"""Dataset and validation metadata contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Base for immutable, strictly typed metadata contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class ColumnRole(StrEnum):
    """Semantic role of a dataset column."""

    FEATURE = "feature"
    TARGET = "target"
    TIMESTAMP = "timestamp"
    ENTITY = "entity"
    GROUP = "group"


class DatasetRef(ContractModel):
    """Stable reference to a source dataset."""

    dataset_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    dataset_version: str | None = None


class DatasetSchema(ContractModel):
    """Canonical dataset schema fingerprint and column roles."""

    fingerprint: str = Field(min_length=1)
    columns: dict[str, str]
    roles: dict[str, ColumnRole] = Field(default_factory=dict)


class ValidationIssue(ContractModel):
    """A row or schema issue found during validation."""

    code: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    message: str = Field(min_length=1)
    rejected_rows: int = Field(default=0, ge=0)


class DatasetManifest(ContractModel):
    """Manifest for an immutable Bronze or Silver dataset artifact."""

    contract_version: str = Field(min_length=1)
    dataset: DatasetRef
    dataset_schema: DatasetSchema = Field(alias="schema")
    source_checksum: str = Field(min_length=1)
    data_checksum: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    accepted_row_count: int | None = Field(default=None, ge=0)
    rejected_row_count: int | None = Field(default=None, ge=0)
    ingestion_config_hash: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    license: str | None = None
    validation_issues: tuple[ValidationIssue, ...] = ()
