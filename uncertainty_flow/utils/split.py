"""Calibration set splitting strategies."""

from abc import ABC, abstractmethod

import polars as pl

from .exceptions import error_calibration_too_small, warn_calibration_size


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
