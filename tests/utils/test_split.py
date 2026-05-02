"""Tests for calibration split strategies."""

import polars as pl
import pytest

from uncertainty_flow.utils import (
    RandomHoldoutSplit,
    TemporalHoldoutSplit,
    select_validation_plan,
)


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


class TestValidationPlanSelector:
    """Test composable validation plan selection."""

    def test_tabular_medium_uses_random_holdout(self):
        df = pl.DataFrame({"x": range(300), "y": range(300)})
        plan = select_validation_plan(df, task_type="tabular", random_state=7)
        assert plan.metadata.strategy_name == "random_holdout"
        assert len(plan.inner_splits) == 0

    def test_tabular_small_uses_cv(self):
        df = pl.DataFrame({"x": range(120), "y": range(120)})
        plan = select_validation_plan(df, task_type="tabular", random_state=7)
        assert plan.metadata.strategy_name == "kfold_cv"
        assert len(plan.inner_splits) >= 2

    def test_time_series_uses_temporal_holdout(self):
        df = pl.DataFrame({"x": range(300), "y": range(300)})
        plan = select_validation_plan(df, task_type="time_series", random_state=7)
        assert plan.metadata.strategy_name == "temporal_holdout"
        train_df, val_df = plan.outer_split
        assert train_df["x"].to_list()[-1] < val_df["x"].to_list()[0]

    def test_hybrid_time_series_has_inner_oos_splits(self):
        df = pl.DataFrame({"x": range(300), "y": range(300)})
        plan = select_validation_plan(
            df,
            task_type="time_series",
            random_state=7,
            hybrid_mode=True,
        )
        assert plan.metadata.strategy_name == "temporal_outer_plus_oos_inner_cv"
        assert len(plan.inner_splits) >= 2

    def test_determinism_same_seed_same_splits(self):
        df = pl.DataFrame({"x": range(120), "y": range(120)})
        plan1 = select_validation_plan(df, task_type="tabular", random_state=11)
        plan2 = select_validation_plan(df, task_type="tabular", random_state=11)
        assert plan1.metadata == plan2.metadata
        assert plan1.outer_split[1]["x"].to_list() == plan2.outer_split[1]["x"].to_list()

    def test_selector_logs_strategy_and_folds(self, caplog):
        caplog.set_level("DEBUG")
        df = pl.DataFrame({"x": range(120), "y": range(120)})
        select_validation_plan(df, task_type="tabular", random_state=7)
        assert "validation_strategy strategy=" in caplog.text
        assert "validation_strategy_fold strategy=" in caplog.text
