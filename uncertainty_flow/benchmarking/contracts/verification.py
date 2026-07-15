"""Machine-readable verification contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Base for immutable verification metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class VerificationStatus(StrEnum):
    """Outcome of one verification check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VerificationSeverity(StrEnum):
    """Impact level of a failed or skipped check."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VerificationCheck(ContractModel):
    """One invariant check and its evidence."""

    check_id: str = Field(min_length=1)
    status: VerificationStatus
    severity: VerificationSeverity
    target: str = Field(min_length=1)
    evidence: dict[str, str] = Field(default_factory=dict)
    failure_message: str | None = None


class RunVerificationReport(ContractModel):
    """Aggregate verification report for a run."""

    run_id: str = Field(min_length=1)
    checks: tuple[VerificationCheck, ...]
    passed: bool
