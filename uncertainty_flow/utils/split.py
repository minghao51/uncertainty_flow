"""Calibration set splitting strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl
from sklearn.model_selection import KFold

from .exceptions import error_calibration_too_small, warn_calibration_size

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BaseSplit(ABC):
    """Base class for calibration split strategies."""

    @abstractmethod
    def split(
        self,
        data: pl.DataFrame,
        calibration_size: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data into (train, calibration) sets.

        Args:
            data: Input DataFrame
            calibration_size: Fraction of data to use for calibration (0-1)

        Returns:
            Tuple of (train_data, calibration_data)

        Raises:
            ValueError: If calibration set would be too small (< 20 samples)
        """
        ...

    def _validate_calibration_size(
        self,
        n_total: int,
        n_calib: int,
    ) -> None:
        """Validate calibration set size."""
        if n_calib < 20:
            error_calibration_too_small(n_calib)
        if n_calib < 50:
            warn_calibration_size(n_calib)


class RandomHoldoutSplit(BaseSplit):
    """Random holdout for tabular data."""

    def __init__(self, random_state: int | None = None):
        """
        Initialize random holdout splitter.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def split(
        self,
        data: pl.DataFrame,
        calibration_size: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data randomly into train and calibration sets.

        Args:
            data: Input DataFrame
            calibration_size: Fraction for calibration (0-1)

        Returns:
            Tuple of (train, calibration) DataFrames
        """
        n_total = len(data)
        n_calib = int(n_total * calibration_size)

        self._validate_calibration_size(n_total, n_calib)

        # Random split
        shuffled = data.sample(fraction=1.0, seed=self.random_state)
        train = shuffled[: n_total - n_calib]
        calib = shuffled[n_total - n_calib :]

        return train, calib


class TemporalHoldoutSplit(BaseSplit):
    """Holdout from END for time series (no data leakage)."""

    def split(
        self,
        data: pl.DataFrame,
        calibration_size: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data temporally, taking last n% for calibration.

        Args:
            data: Input DataFrame (assumed to be temporally ordered)
            calibration_size: Fraction for calibration (0-1)

        Returns:
            Tuple of (train, calibration) DataFrames
        """
        n_total = len(data)
        n_calib = int(n_total * calibration_size)

        self._validate_calibration_size(n_total, n_calib)

        # Take LAST n% for calibration (temporal ordering)
        train = data[: n_total - n_calib]
        calib = data[n_total - n_calib :]

        return train, calib


@dataclass(frozen=True)
class SplitPlanMetadata:
    """Metadata describing how validation splits were selected."""

    strategy_name: str
    reason: str
    n_samples: int
    n_splits: int
    holdout_fraction: float
    random_state: int | None
    hybrid_mode: bool
    task_type: Literal["tabular", "time_series"]


@dataclass(frozen=True)
class ValidationSplitPlan:
    """Composable split plan with required outer split and optional inner splits."""

    outer_split: tuple[pl.DataFrame, pl.DataFrame]
    inner_splits: list[tuple[pl.DataFrame, pl.DataFrame]]
    metadata: SplitPlanMetadata


def _build_random_holdout(
    data: pl.DataFrame,
    holdout_fraction: float,
    random_state: int | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    splitter = RandomHoldoutSplit(random_state=random_state)
    return splitter.split(data, holdout_fraction)


def _build_temporal_holdout(
    data: pl.DataFrame,
    holdout_fraction: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    splitter = TemporalHoldoutSplit()
    return splitter.split(data, holdout_fraction)


def _build_kfold_splits(
    data: pl.DataFrame,
    n_splits: int,
    random_state: int | None,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    indexed = data.with_row_index("__row_idx__")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    for train_idx, val_idx in kfold.split(range(len(data))):
        train_df = indexed.filter(pl.col("__row_idx__").is_in(list(train_idx))).drop("__row_idx__")
        val_df = indexed.filter(pl.col("__row_idx__").is_in(list(val_idx))).drop("__row_idx__")
        splits.append((train_df, val_df))
    return splits


def select_validation_plan(
    data: pl.DataFrame,
    *,
    task_type: Literal["tabular", "time_series"],
    random_state: int | None = None,
    holdout_fraction: float = 0.2,
    small_data_threshold: int = 250,
    cv_splits: int = 5,
    hybrid_mode: bool = False,
    enable_logging: bool = True,
) -> ValidationSplitPlan:
    """Select a deterministic validation split plan for tuning/evaluation.

    Hybrid mode means:
    - time_series: temporal outer split + random out-of-sample inner split(s) on outer-train
    - tabular: random outer split + random out-of-sample inner split(s) on outer-train
    """
    n_samples = len(data)
    n_splits = 1

    if task_type == "time_series":
        outer_train, outer_val = _build_temporal_holdout(data, holdout_fraction)
        inner_splits: list[tuple[pl.DataFrame, pl.DataFrame]] = []
        strategy_name = "temporal_holdout"
        reason = "time_series task defaults to temporal holdout"
        if hybrid_mode:
            inner_splits = _build_kfold_splits(
                outer_train,
                n_splits=min(cv_splits, max(2, len(outer_train) // 20)),
                random_state=random_state,
            )
            n_splits = len(inner_splits)
            strategy_name = "temporal_outer_plus_oos_inner_cv"
            reason = "hybrid mode enabled: temporal outer split with out-of-sample inner CV"
    else:
        outer_train, outer_val = _build_random_holdout(data, holdout_fraction, random_state)
        inner_splits = []
        strategy_name = "random_holdout"
        reason = "tabular task defaults to random holdout"
        if n_samples <= small_data_threshold:
            inner_splits = _build_kfold_splits(
                data,
                n_splits=min(cv_splits, max(2, n_samples // 20)),
                random_state=random_state,
            )
            n_splits = len(inner_splits)
            strategy_name = "kfold_cv"
            reason = "small tabular dataset uses CV for more stable tuning"
        if hybrid_mode:
            inner_splits = _build_kfold_splits(
                outer_train,
                n_splits=min(cv_splits, max(2, len(outer_train) // 20)),
                random_state=random_state,
            )
            n_splits = len(inner_splits)
            strategy_name = "random_outer_plus_oos_inner_cv"
            reason = "hybrid mode enabled: out-of-sample outer and inner validation"

    metadata = SplitPlanMetadata(
        strategy_name=strategy_name,
        reason=reason,
        n_samples=n_samples,
        n_splits=n_splits,
        holdout_fraction=holdout_fraction,
        random_state=random_state,
        hybrid_mode=hybrid_mode,
        task_type=task_type,
    )

    if enable_logging:
        logger.info(
            "validation_strategy strategy=%s reason=%s task_type=%s n_samples=%d n_splits=%d "
            "holdout_fraction=%.3f random_state=%s hybrid_mode=%s",
            metadata.strategy_name,
            metadata.reason,
            metadata.task_type,
            metadata.n_samples,
            metadata.n_splits,
            metadata.holdout_fraction,
            metadata.random_state,
            metadata.hybrid_mode,
        )
        for idx, (train_df, val_df) in enumerate(inner_splits, start=1):
            logger.debug(
                "validation_strategy_fold strategy=%s fold=%d train_rows=%d val_rows=%d",
                metadata.strategy_name,
                idx,
                len(train_df),
                len(val_df),
            )

    return ValidationSplitPlan(
        outer_split=(outer_train, outer_val),
        inner_splits=inner_splits,
        metadata=metadata,
    )
