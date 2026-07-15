"""Validation-plan and persisted split contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Base for immutable validation metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SplitStrategy(StrEnum):
    """Supported initial split strategies."""

    RANDOM_HOLDOUT = "random_holdout"
    TEMPORAL_HOLDOUT = "temporal_holdout"
    ROLLING_ORIGIN = "rolling_origin"


class SplitAssignment(ContractModel):
    """Persisted membership for one observation."""

    observation_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    fold: int | None = Field(default=None, ge=0)


class LeakageCheckResult(ContractModel):
    """Result of a cross-split leakage check."""

    passed: bool
    checked_rows: int = Field(ge=0)
    violations: tuple[str, ...] = ()


class ValidationPlan(ContractModel):
    """Complete immutable description of a dataset evaluation population."""

    validation_plan_id: str = Field(min_length=1)
    strategy: SplitStrategy
    random_seed: int | None = None
    test_size: float | None = Field(default=None, gt=0, lt=1)
    calibration_size: float | None = Field(default=None, gt=0, lt=1)
    temporal_cutoffs: tuple[str, ...] = ()
    assignments: tuple[SplitAssignment, ...]
    leakage_check: LeakageCheckResult

    @model_validator(mode="after")
    def validate_assignments(self) -> ValidationPlan:
        if self.strategy != SplitStrategy.ROLLING_ORIGIN:
            ids = [assignment.observation_id for assignment in self.assignments]
            if len(ids) != len(set(ids)):
                raise ValueError("Validation plan contains duplicate observation assignments")
            if any(assignment.fold is not None for assignment in self.assignments):
                raise ValueError("Non-rolling validation assignments cannot have fold IDs")
        else:
            folds = {assignment.fold for assignment in self.assignments}
            if None in folds or not folds:
                raise ValueError("Rolling-origin assignments must include fold IDs")
            for fold in sorted(fold for fold in folds if fold is not None):
                fold_assignments = [
                    assignment for assignment in self.assignments if assignment.fold == fold
                ]
                train_ids = {
                    assignment.observation_id
                    for assignment in fold_assignments
                    if assignment.split == "train"
                }
                test_ids = {
                    assignment.observation_id
                    for assignment in fold_assignments
                    if assignment.split == "test"
                }
                if not train_ids or not test_ids or not train_ids.isdisjoint(test_ids):
                    raise ValueError(f"Rolling-origin fold {fold} has invalid membership")
        if not self.assignments:
            raise ValueError("Validation plan must contain at least one split assignment")
        return self
