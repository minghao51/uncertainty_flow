"""Tests for calibration split strategies."""

import polars as pl
import pytest

from uncertainty_flow.utils import (
    RandomHoldoutSplit,
    RollingOriginSplit,
    SlidingWindowSplit,
    TemporalHoldoutSplit,
    rolling_origin_splits,
    select_validation_plan,
)
from uncertainty_flow.utils.exceptions import CalibrationSizeError, UncertaintyFlowWarning


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
        with pytest.warns(UncertaintyFlowWarning, match="Calibration set contains only"):
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
        with pytest.warns(UncertaintyFlowWarning, match="Calibration set contains only"):
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

    def test_rolling_origin_time_series(self):
        df = pl.DataFrame({"x": range(200), "y": range(200)})
        plan = select_validation_plan(
            df,
            task_type="time_series",
            rolling_origin=True,
            rolling_min_train=50,
            rolling_horizon=5,
        )
        assert plan.metadata.strategy_name == "rolling_origin"
        assert len(plan.inner_splits) >= 2
        for train_df, val_df in plan.inner_splits:
            assert len(train_df) >= 50
            assert len(val_df) == 5


class TestRollingOriginSplit:
    def test_expanding_window(self):
        df = pl.DataFrame({"x": range(100), "y": range(100)})
        splitter = RollingOriginSplit(n_splits=3, min_train_size=30, horizon=5)
        folds = splitter.splits(df)
        assert len(folds) == 3

        for train, test in folds:
            assert len(train) >= 30
            assert len(test) == 5

        assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])

    def test_no_overlap_between_train_test(self):
        df = pl.DataFrame({"x": range(100), "y": range(100)})
        splitter = RollingOriginSplit(n_splits=3, min_train_size=30, horizon=5)
        folds = splitter.splits(df)
        for train, test in folds:
            train_max = train["x"].max()
            test_min = test["x"].min()
            assert train_max < test_min

    def test_gap_between_train_test(self):
        df = pl.DataFrame({"x": range(100), "y": range(100)})
        splitter = RollingOriginSplit(n_splits=2, min_train_size=30, horizon=5, gap=3)
        folds = splitter.splits(df)
        for train, test in folds:
            train_max = train["x"].max()
            test_min = test["x"].min()
            assert test_min - train_max > 1

    def test_data_too_short_raises(self):
        df = pl.DataFrame({"x": range(10), "y": range(10)})
        splitter = RollingOriginSplit(n_splits=2, min_train_size=50, horizon=5)
        with pytest.raises(ValueError, match="too short"):
            splitter.splits(df)

    def test_convenience_function(self):
        df = pl.DataFrame({"x": range(100), "y": range(100)})
        folds = rolling_origin_splits(df, n_splits=3, min_train_size=30, horizon=5)
        assert len(folds) == 3


class TestSlidingWindowSplit:
    def test_fixed_window_size(self):
        df = pl.DataFrame({"x": range(200), "y": range(200)})
        splitter = SlidingWindowSplit(n_splits=3, train_size=50, horizon=5)
        folds = splitter.splits(df)
        assert len(folds) == 3

        for train, test in folds:
            assert len(train) == 50
            assert len(test) == 5

    def test_windows_slide_forward(self):
        df = pl.DataFrame({"x": range(200), "y": range(200)})
        splitter = SlidingWindowSplit(n_splits=3, train_size=50, horizon=5, step=10)
        folds = splitter.splits(df)
        first_start = folds[0][0]["x"].min()
        second_start = folds[1][0]["x"].min()
        assert second_start - first_start == 10

    def test_no_overlap_train_test(self):
        df = pl.DataFrame({"x": range(200), "y": range(200)})
        splitter = SlidingWindowSplit(n_splits=3, train_size=50, horizon=5)
        folds = splitter.splits(df)
        for train, test in folds:
            train_max = train["x"].max()
            test_min = test["x"].min()
            assert train_max < test_min

    def test_data_too_short_raises(self):
        df = pl.DataFrame({"x": range(20), "y": range(20)})
        splitter = SlidingWindowSplit(n_splits=2, train_size=50, horizon=5)
        with pytest.raises(ValueError, match="too short"):
            splitter.splits(df)


class TestRollingOriginSplitValidation:
    def test_rejects_negative_n_splits(self):
        with pytest.raises(ValueError, match="n_splits must be"):
            RollingOriginSplit(n_splits=0)

    def test_rejects_zero_min_train_size(self):
        with pytest.raises(ValueError, match="min_train_size"):
            RollingOriginSplit(min_train_size=0)

    def test_rejects_negative_horizon(self):
        with pytest.raises(ValueError, match="horizon must be"):
            RollingOriginSplit(horizon=0)

    def test_rejects_negative_gap(self):
        with pytest.raises(ValueError, match="gap must be"):
            RollingOriginSplit(gap=-1)

    def test_requested_more_splits_than_available(self):
        df = pl.DataFrame({"x": range(60), "y": range(60)})
        splitter = RollingOriginSplit(n_splits=10, min_train_size=50, horizon=5)
        with pytest.raises(ValueError, match="Requested 10 splits"):
            splitter.splits(df)

    def test_fold_beyond_data_raises(self):
        df = pl.DataFrame({"x": range(60), "y": range(60)})
        splitter = RollingOriginSplit(n_splits=2, min_train_size=55, horizon=5)
        with pytest.raises(ValueError, match="Requested 2 splits"):
            splitter.splits(df)


class TestSlidingWindowSplitValidation:
    def test_rejects_negative_n_splits(self):
        with pytest.raises(ValueError, match="n_splits must be"):
            SlidingWindowSplit(n_splits=0)

    def test_rejects_zero_train_size(self):
        with pytest.raises(ValueError, match="train_size must be"):
            SlidingWindowSplit(train_size=0)

    def test_rejects_negative_horizon(self):
        with pytest.raises(ValueError, match="horizon must be"):
            SlidingWindowSplit(horizon=0)

    def test_rejects_negative_gap(self):
        with pytest.raises(ValueError, match="gap must be"):
            SlidingWindowSplit(gap=-1)

    def test_fold_beyond_data_raises(self):
        df = pl.DataFrame({"x": range(200), "y": range(200)})
        splitter = SlidingWindowSplit(n_splits=5, train_size=100, horizon=50, step=30)
        with pytest.raises(ValueError, match="extends beyond"):
            splitter.splits(df)


class TestValidationPlanEdgeCases:
    def test_tabular_hybrid_adds_inner_splits(self):
        df = pl.DataFrame({"x": range(300), "y": range(300)})
        plan = select_validation_plan(df, task_type="tabular", random_state=7, hybrid_mode=True)
        assert plan.metadata.strategy_name == "random_outer_plus_oos_inner_cv"
        assert len(plan.inner_splits) >= 2

    def test_time_series_rolling_not_enough_data(self):
        df = pl.DataFrame({"x": range(100), "y": range(100)})
        plan = select_validation_plan(
            df,
            task_type="time_series",
            rolling_origin=True,
            rolling_min_train=90,
            rolling_horizon=20,
        )
        assert plan.metadata.strategy_name == "temporal_holdout"
        assert len(plan.inner_splits) == 0

    def test_time_series_rolling_with_enough_data(self):
        df = pl.DataFrame({"x": range(300), "y": range(300)})
        plan = select_validation_plan(
            df,
            task_type="time_series",
            rolling_origin=True,
            rolling_min_train=50,
            rolling_horizon=5,
        )
        assert plan.metadata.strategy_name == "rolling_origin"

    def test_calibration_size_error_message(self):
        splitter = RandomHoldoutSplit(random_state=42)
        df = pl.DataFrame({"x": range(30), "y": range(30)})
        with pytest.raises(CalibrationSizeError, match="Calibration set too small"):
            splitter.split(df, 0.5)

    def test_select_plan_does_not_log_when_disabled(self):
        df = pl.DataFrame({"x": range(120), "y": range(120)})
        plan = select_validation_plan(
            df,
            task_type="tabular",
            random_state=7,
            enable_logging=False,
        )
        assert plan.metadata.n_samples == 120
