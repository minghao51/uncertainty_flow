"""Calibration set splitting strategies."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import polars as pl
from sklearn.model_selection import KFold

from .exceptions import CalibrationSizeError, UncertaintyFlowWarning

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
            raise CalibrationSizeError(n_calib)
        if n_calib < 50:
            warnings.warn(
                f"Calibration set contains only {n_calib} samples. "
                "Consider increasing calibration size for more stable uncertainty "
                "estimates. [UF-W001]",
                UncertaintyFlowWarning,
                stacklevel=3,
            )


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


class RollingOriginSplit:
    """Expanding-window (rolling-origin) split for time series evaluation.

    Each fold uses all data up to an origin point as training and the next
    ``horizon`` rows as the test set. The origin advances by ``step`` rows
    each fold, producing an expanding training window.

    Args:
        n_splits: Number of folds.
        min_train_size: Minimum number of rows in the first training window.
        horizon: Number of rows in each test set.
        gap: Number of rows between train end and test start (default 0).
        step: How far the origin advances per fold. Defaults to ``horizon``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 50,
        horizon: int = 1,
        gap: int = 0,
        step: int | None = None,
    ):
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")

        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.gap = gap
        self.step = step if step is not None else horizon

    def splits(
        self,
        data: pl.DataFrame,
    ) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate expanding-window (train, test) pairs.

        Args:
            data: DataFrame assumed to be in temporal order.

        Returns:
            List of (train_df, test_df) tuples.

        Raises:
            ValueError: If data is too short for the requested configuration.
        """
        n = len(data)
        last_train_end = n - self.gap - self.horizon
        first_train_end = self.min_train_size - 1

        if first_train_end > last_train_end:
            raise ValueError(
                f"Data too short for RollingOriginSplit: need at least "
                f"{self.min_train_size + self.gap + self.horizon} rows, got {n}"
            )

        available_folds = (last_train_end - first_train_end) // self.step + 1
        if self.n_splits > available_folds:
            raise ValueError(
                f"Requested {self.n_splits} splits but only {available_folds} "
                f"fit in data of length {n} with min_train_size={self.min_train_size}, "
                f"horizon={self.horizon}, gap={self.gap}"
            )

        origin = first_train_end
        result: list[tuple[pl.DataFrame, pl.DataFrame]] = []
        for _ in range(self.n_splits):
            train_end = origin + 1
            test_start = origin + 1 + self.gap
            test_end = test_start + self.horizon

            if test_end > n:
                raise ValueError(f"Fold extends beyond data: test_end={test_end} > n={n}")

            result.append((data[:train_end], data[test_start:test_end]))
            origin += self.step

        return result


class SlidingWindowSplit:
    """Fixed-width sliding-window split for time series evaluation.

    Each fold uses a training window of fixed ``train_size`` rows that slides
    forward by ``step`` rows each fold.

    Args:
        n_splits: Number of folds.
        train_size: Number of rows in each training window.
        horizon: Number of rows in each test set.
        gap: Number of rows between train end and test start (default 0).
        step: How far the window advances per fold. Defaults to ``horizon``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 100,
        horizon: int = 1,
        gap: int = 0,
        step: int | None = None,
    ):
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {train_size}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")

        self.n_splits = n_splits
        self.train_size = train_size
        self.horizon = horizon
        self.gap = gap
        self.step = step if step is not None else horizon

    def splits(
        self,
        data: pl.DataFrame,
    ) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate fixed-window (train, test) pairs.

        Args:
            data: DataFrame assumed to be in temporal order.

        Returns:
            List of (train_df, test_df) tuples.

        Raises:
            ValueError: If data is too short for the requested configuration.
        """
        n = len(data)
        if n < self.train_size + self.gap + self.horizon:
            raise ValueError(
                f"Data too short for SlidingWindowSplit: need at least "
                f"{self.train_size + self.gap + self.horizon} rows, got {n}"
            )

        result: list[tuple[pl.DataFrame, pl.DataFrame]] = []
        for i in range(self.n_splits):
            train_start = i * self.step
            train_end = train_start + self.train_size
            test_start = train_end + self.gap
            test_end = test_start + self.horizon

            if test_end > n:
                raise ValueError(
                    f"Fold {i} extends beyond data: test_end={test_end} > n={n}. "
                    f"Reduce n_splits or step."
                )

            result.append((data[train_start:train_end], data[test_start:test_end]))

        return result


def rolling_origin_splits(
    data: pl.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 50,
    horizon: int = 1,
    gap: int = 0,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Convenience function for rolling-origin (expanding window) splits.

    Args:
        data: DataFrame in temporal order.
        n_splits: Number of folds.
        min_train_size: Minimum training rows in the first fold.
        horizon: Test set size per fold.
        gap: Rows between train end and test start.

    Returns:
        List of (train_df, test_df) tuples.
    """
    splitter = RollingOriginSplit(
        n_splits=n_splits,
        min_train_size=min_train_size,
        horizon=horizon,
        gap=gap,
    )
    return splitter.splits(data)


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
    rolling_origin: bool = False,
    rolling_min_train: int = 50,
    rolling_horizon: int = 1,
) -> ValidationSplitPlan:
    """Select a deterministic validation split plan for tuning/evaluation.

    Hybrid mode means:
    - time_series: temporal outer split + random out-of-sample inner split(s) on outer-train
    - tabular: random outer split + random out-of-sample inner split(s) on outer-train

    When ``rolling_origin=True`` and ``task_type="time_series"``, the outer
    split uses a single temporal holdout (as before) and the inner splits
    use :class:`RollingOriginSplit` instead of random K-fold.
    """
    n_samples = len(data)
    n_splits = 1

    if task_type == "time_series":
        outer_train, outer_val = _build_temporal_holdout(data, holdout_fraction)
        inner_splits: list[tuple[pl.DataFrame, pl.DataFrame]] = []
        strategy_name = "temporal_holdout"
        reason = "time_series task defaults to temporal holdout"

        if rolling_origin and len(outer_train) >= rolling_min_train + rolling_horizon:
            n_avail = (len(outer_train) - rolling_min_train) // rolling_horizon
            inner_splits = RollingOriginSplit(
                n_splits=min(cv_splits, max(2, n_avail)),
                min_train_size=rolling_min_train,
                horizon=rolling_horizon,
            ).splits(outer_train)
            n_splits = len(inner_splits)
            strategy_name = "rolling_origin"
            reason = "time_series task with rolling-origin evaluation"
        elif hybrid_mode:
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
                outer_train,
                n_splits=min(cv_splits, max(2, len(outer_train) // 20)),
                random_state=random_state,
            )
            n_splits = len(inner_splits)
            strategy_name = "kfold_cv"
            reason = "small tabular dataset uses CV for more stable tuning"
        elif hybrid_mode:
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
