"""Tests for risk functions."""

import numpy as np

from uncertainty_flow.risk.risk_functions import (
    asymmetric_loss,
    financial_var,
    inventory_cost,
    threshold_penalty,
)


class TestAsymmetricLoss:
    """Test asymmetric_loss function."""

    def test_returns_callable(self):
        """Should return a callable risk function."""
        risk_fn = asymmetric_loss()
        assert callable(risk_fn)

    def test_overprediction_penalty(self):
        """Should penalize overpredictions correctly."""
        risk_fn = asymmetric_loss(overprediction_penalty=1.0, underprediction_penalty=2.0)
        y_true = np.array([10.0])
        y_pred = np.array([12.0])  # overprediction by 2
        risk = risk_fn(y_true, y_pred)
        assert risk[0] == 2.0  # 1.0 * 2

    def test_underprediction_penalty(self):
        """Should penalize underpredictions correctly."""
        risk_fn = asymmetric_loss(overprediction_penalty=1.0, underprediction_penalty=2.0)
        y_true = np.array([10.0])
        y_pred = np.array([8.0])  # underprediction by 2
        risk = risk_fn(y_true, y_pred)
        assert risk[0] == 4.0  # 2.0 * 2

    def test_custom_penalties(self):
        """Should respect custom penalty values."""
        risk_fn = asymmetric_loss(overprediction_penalty=3.0, underprediction_penalty=5.0)
        y_true = np.array([10.0])
        y_pred = np.array([12.0])
        risk = risk_fn(y_true, y_pred)
        assert risk[0] == 6.0  # 3.0 * 2


class TestThresholdPenalty:
    """Test threshold_penalty function."""

    def test_returns_callable(self):
        """Should return a callable risk function."""
        risk_fn = threshold_penalty(threshold=5.0)
        assert callable(risk_fn)

    def test_below_threshold(self):
        """Should apply base penalty when error is below threshold."""
        risk_fn = threshold_penalty(threshold=5.0, penalty_above=10.0, penalty_below=1.0)
        y_true = np.array([100.0])
        y_pred = np.array([103.0])  # error of 3, below threshold
        risk = risk_fn(y_true, y_pred)
        assert risk[0] == 3.0  # 1.0 * 3

    def test_above_threshold(self):
        """Should apply escalated penalty when error exceeds threshold."""
        risk_fn = threshold_penalty(threshold=5.0, penalty_above=10.0, penalty_below=1.0)
        y_true = np.array([100.0])
        y_pred = np.array([110.0])  # error of 10, 5 above threshold
        risk = risk_fn(y_true, y_pred)
        assert risk[0] == 50.0  # 10.0 * (10 - 5)

    def test_custom_threshold(self):
        """Should respect custom threshold value."""
        risk_fn = threshold_penalty(threshold=3.0, penalty_above=10.0, penalty_below=1.0)
        y_true = np.array([100.0])
        y_pred = np.array([105.0])  # error of 5, 2 above threshold
        risk = risk_fn(y_true, y_pred)
        assert risk[0] == 20.0  # 10.0 * (5 - 3)


class TestInventoryCost:
    """Test inventory_cost function."""

    def test_returns_callable(self):
        """Should return a callable risk function."""
        risk_fn = inventory_cost()
        assert callable(risk_fn)

    def test_overprediction_holding_cost(self):
        """Should charge holding cost for overpredicted demand."""
        risk_fn = inventory_cost(holding_cost=1.0, stockout_cost=10.0)
        demand = np.array([100.0])
        forecast = np.array([110.0])  # overforecast by 10
        cost = risk_fn(demand, forecast)
        assert cost[0] == 10.0  # 1.0 * 10

    def test_underprediction_stockout_cost(self):
        """Should charge stockout cost for underpredicted demand."""
        risk_fn = inventory_cost(holding_cost=1.0, stockout_cost=10.0)
        demand = np.array([100.0])
        forecast = np.array([90.0])  # underforecast by 10
        cost = risk_fn(demand, forecast)
        assert cost[0] == 100.0  # 10.0 * 10

    def test_custom_costs(self):
        """Should respect custom cost values."""
        risk_fn = inventory_cost(holding_cost=2.0, stockout_cost=5.0)
        demand = np.array([100.0])
        forecast = np.array([110.0])  # overforecast by 10
        cost = risk_fn(demand, forecast)
        assert cost[0] == 20.0  # 2.0 * 10


class TestFinancialVar:
    """Test financial_var function."""

    def test_returns_callable(self):
        """Should return a callable risk function."""
        risk_fn = financial_var()
        assert callable(risk_fn)

    def test_computes_var_threshold(self):
        """Should compute VaR threshold from losses."""
        risk_fn = financial_var(var_level=0.95)
        y_true = np.array([100.0] * 20)
        # Varying predictions to create different loss levels
        y_pred = np.array([100.0 + i for i in range(20)])
        risk = risk_fn(y_true, y_pred)
        # Risk values should be non-negative
        assert (risk >= 0).all()

    def test_custom_var_level(self):
        """Should respect custom VaR level."""
        risk_fn = financial_var(var_level=0.9)
        y_true = np.array([100.0] * 20)
        y_pred = np.array([100.0 + i for i in range(20)])
        risk = risk_fn(y_true, y_pred)
        # Risk values should be non-negative
        assert (risk >= 0).all()


class TestRiskFunctionsIntegration:
    """Integration tests for risk functions."""

    def test_all_risk_functions_return_arrays(self):
        """All risk functions should return numpy arrays."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])

        risk_fns = [
            asymmetric_loss(),
            threshold_penalty(threshold=5.0),
            inventory_cost(),
            financial_var(),
        ]

        for risk_fn in risk_fns:
            risk = risk_fn(y_true, y_pred)
            assert isinstance(risk, np.ndarray)
            assert len(risk) == len(y_true)

    def test_all_risk_functions_non_negative(self):
        """All risk functions should return non-negative values."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])

        risk_fns = [
            asymmetric_loss(),
            threshold_penalty(threshold=5.0),
            inventory_cost(),
            financial_var(),
        ]

        for risk_fn in risk_fns:
            risk = risk_fn(y_true, y_pred)
            assert (risk >= 0).all()
