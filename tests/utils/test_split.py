"""Tests for calibration split strategies."""

import polars as pl
import pytest

from uncertainty_flow.utils import RandomHoldoutSplit, TemporalHoldoutSplit


class TestRandomHoldoutSplit:
    """Test random holdout split."""

    def test_split_returns_two_dataframes(self):
        """Should return train and calibration DataFrames."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter = RandomHoldoutSplit(random_state=42)
        train, calib = splitter.split(df, 0.2)
        assert isinstance(train, pl.DataFrame)
        assert isinstance(calib, pl.DataFrame)

    def test_split_sizes(self):
        """Should create calibration set of specified size."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter = RandomHoldoutSplit(random_state=42)
        train, calib = splitter.split(df, 0.2)
        assert len(calib) == 20  # 20% of 100
        assert len(train) == 80

    def test_reproducibility_with_random_state(self):
        """Should be reproducible with same random state."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter1 = RandomHoldoutSplit(random_state=42)
        splitter2 = RandomHoldoutSplit(random_state=42)
        train1, calib1 = splitter1.split(df, 0.2)
        train2, calib2 = splitter2.split(df, 0.2)
        # Should be identical
        assert calib1["a"].to_list() == calib2["a"].to_list()

    def test_raises_error_too_small_calibration(self):
        """Should raise error if calibration set < 20 samples."""
        df = pl.DataFrame(
            {
                "a": range(50),
                "b": range(50, 100),
            }
        )
        splitter = RandomHoldoutSplit(random_state=42)
        with pytest.raises(ValueError, match="Calibration set too small"):
            splitter.split(df, 0.3)  # Would give 15 samples

    def test_warns_small_calibration(self):
        """Should warn if calibration set < 50 samples."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter = RandomHoldoutSplit(random_state=42)
        with pytest.warns(UserWarning, match="Calibration set has only"):
            splitter.split(df, 0.4)  # 40 samples


class TestTemporalHoldoutSplit:
    """Test temporal holdout split."""

    def test_split_from_end(self):
        """Should take last n% for calibration."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter = TemporalHoldoutSplit()
        train, calib = splitter.split(df, 0.2)

        # Calibration should be last 20 rows
        assert calib["a"].to_list() == list(range(80, 100))
        # Train should be first 80 rows
        assert train["a"].to_list() == list(range(80))

    def test_split_sizes(self):
        """Should create calibration set of specified size."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter = TemporalHoldoutSplit()
        train, calib = splitter.split(df, 0.2)
        assert len(calib) == 20
        assert len(train) == 80

    def test_no_shuffling(self):
        """Should not shuffle data (temporal ordering preserved)."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter = TemporalHoldoutSplit()
        train, calib = splitter.split(df, 0.2)

        # Check that order is preserved
        assert train["a"].to_list() == sorted(train["a"].to_list())
        assert calib["a"].to_list() == sorted(calib["a"].to_list())

    def test_raises_error_too_small_calibration(self):
        """Should raise error if calibration set < 20 samples."""
        df = pl.DataFrame(
            {
                "a": range(50),
                "b": range(50, 100),
            }
        )
        splitter = TemporalHoldoutSplit()
        with pytest.raises(ValueError, match="Calibration set too small"):
            splitter.split(df, 0.3)  # Would give 15 samples

    def test_warns_small_calibration(self):
        """Should warn if calibration set < 50 samples."""
        df = pl.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
            }
        )
        splitter = TemporalHoldoutSplit()
        with pytest.warns(UserWarning, match="Calibration set has only"):
            splitter.split(df, 0.4)  # 40 samples
