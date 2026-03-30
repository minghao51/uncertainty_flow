"""Tests for pinball_loss metric."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.metrics import pinball_loss


class TestPinballLoss:
    """Test pinball loss calculation."""

    def test_numpy_arrays(self):
        """Should work with numpy arrays."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.5])
        loss = pinball_loss(y_true, y_pred, 0.5)
        # For median (0.5), loss = 0.5 * MAE = 0.5 * 0.5 = 0.25
        assert loss == 0.25

    def test_polars_series(self):
        """Should work with polars Series."""
        y_true = pl.Series([1, 2, 3, 4, 5])
        y_pred = pl.Series([1.5, 2.5, 2.5, 4.5, 4.5])
        loss = pinball_loss(y_true, y_pred, 0.5)
        assert loss == 0.25

    def test_upper_quantile_penalizes_underprediction(self):
        """Upper quantile should penalize under-prediction more."""
        y_true = np.array([10, 10, 10])
        y_pred = np.array([5, 5, 5])  # Under-predict

        # For 0.9 quantile, under-prediction is penalized heavily
        loss_high = pinball_loss(y_true, y_pred, 0.9)
        loss_low = pinball_loss(y_true, y_pred, 0.1)

        assert loss_high > loss_low

    def test_lower_quantile_penalizes_overprediction(self):
        """Lower quantile should penalize over-prediction more."""
        y_true = np.array([10, 10, 10])
        y_pred = np.array([15, 15, 15])  # Over-predict

        # For 0.1 quantile, over-prediction is penalized heavily
        loss_low = pinball_loss(y_true, y_pred, 0.1)
        loss_high = pinball_loss(y_true, y_pred, 0.9)

        assert loss_low > loss_high

    def test_validates_quantile_range(self):
        """Should raise error for invalid quantile."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="quantile must be in \\(0, 1\\)"):
            pinball_loss(y_true, y_pred, 1.5)

        with pytest.raises(ValueError, match="quantile must be in \\(0, 1\\)"):
            pinball_loss(y_true, y_pred, -0.1)

    def test_perfect_prediction_zero_loss(self):
        """Perfect prediction should have zero loss."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        loss = pinball_loss(y_true, y_pred, 0.5)
        assert loss == 0

    def test_median_is_absolute_error(self):
        """Median quantile (0.5) should equal 0.5 * MAE."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        loss = pinball_loss(y_true, y_pred, 0.5)
        # For quantile 0.5, loss = 0.5 * MAE
        mae = np.mean(np.abs(y_true - y_pred))
        assert loss == 0.5 * mae
