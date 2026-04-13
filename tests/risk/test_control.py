"""Tests for ConformalRiskControl."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.risk import ConformalRiskControl, asymmetric_loss
from uncertainty_flow.wrappers import ConformalRegressor


class TestConformalRiskControlInit:
    """Test ConformalRiskControl initialization."""

    def test_init_with_required_params(self):
        """Should initialize with required parameters."""
        base_model = GradientBoostingRegressor(random_state=42)
        risk_fn = asymmetric_loss()
        risk_control = ConformalRiskControl(base_model, risk_fn)
        assert risk_control.base_model is base_model
        assert risk_control.risk_function is risk_fn
        assert risk_control.target_risk == 0.1

    def test_init_with_custom_params(self):
        """Should accept custom parameters."""
        base_model = GradientBoostingRegressor(random_state=42)
        risk_fn = asymmetric_loss()
        risk_control = ConformalRiskControl(
            base_model,
            risk_fn,
            target_risk=0.05,
            calibration_method="mean",
            random_state=42,
        )
        assert risk_control.target_risk == 0.05
        assert risk_control.calibration_method == "mean"
        assert risk_control.random_state == 42


class TestConformalRiskControlFit:
    """Test fit method."""

    def test_fit_returns_self(self, sample_model, risk_control, sample_data):
        """Should return self for method chaining."""
        result = risk_control.fit(sample_data, target="y")
        assert result is risk_control

    def test_fit_sets_risk_threshold(self, sample_model, risk_control, sample_data):
        """Should set risk_threshold after fitting."""
        risk_control.fit(sample_data, target="y")
        assert risk_control._risk_threshold is not None

    def test_fit_sets_calibration_risks(self, sample_model, risk_control, sample_data):
        """Should store calibration risks."""
        risk_control.fit(sample_data, target="y")
        assert risk_control._calibration_risks is not None
        assert len(risk_control._calibration_risks) == sample_data.height

    def test_fit_with_invalid_calibration_method_raises(self, sample_model, sample_data):
        """Should raise ValueError for invalid calibration method."""
        risk_fn = asymmetric_loss()
        risk_control = ConformalRiskControl(
            sample_model,
            risk_fn,
            calibration_method="invalid",
        )
        with pytest.raises(ValueError, match="Unknown calibration_method"):
            risk_control.fit(sample_data, target="y")


class TestConformalRiskControlPredict:
    """Test predict method."""

    def test_predict_before_fit_raises(self, sample_model):
        """Should raise error if predict called before fit."""
        risk_fn = asymmetric_loss()
        risk_control = ConformalRiskControl(sample_model, risk_fn)
        test_data = pl.DataFrame({"x1": [1, 2], "x2": [3, 4]})
        with pytest.raises(Exception):  # InvalidDataError
            risk_control.predict(test_data)

    def test_predict_returns_dataframe(self, sample_model, risk_control, sample_data):
        """Should return a Polars DataFrame."""
        risk_control.fit(sample_data, target="y")
        result = risk_control.predict(sample_data)
        assert isinstance(result, pl.DataFrame)

    def test_predict_has_expected_columns(self, sample_model, risk_control, sample_data):
        """Should have expected columns."""
        risk_control.fit(sample_data, target="y")
        result = risk_control.predict(sample_data)
        expected_cols = ["prediction", "risk", "exceeds_threshold"]
        for col in expected_cols:
            assert col in result.columns

    def test_predict_row_count_matches_input(self, sample_model, risk_control, sample_data):
        """Should have same number of rows as input."""
        risk_control.fit(sample_data, target="y")
        test_data = sample_data.head(50)
        result = risk_control.predict(test_data)
        assert result.height == 50


class TestConformalRiskControlRiskThreshold:
    """Test risk_threshold method."""

    def test_risk_threshold_before_fit_raises(self, sample_model):
        """Should raise error if called before fit."""
        risk_fn = asymmetric_loss()
        risk_control = ConformalRiskControl(sample_model, risk_fn)
        with pytest.raises(Exception):  # InvalidDataError
            risk_control.risk_threshold()

    def test_risk_threshold_after_fit_returns_value(self, sample_model, risk_control, sample_data):
        """Should return threshold value after fitting."""
        risk_control.fit(sample_data, target="y")
        threshold = risk_control.risk_threshold()
        assert isinstance(threshold, float)
        assert threshold >= 0


class TestConformalRiskControlSummary:
    """Test summary method."""

    def test_summary_returns_dict(self, risk_control):
        """Should return a dictionary."""
        summary = risk_control.summary()
        assert isinstance(summary, dict)

    def test_summary_has_expected_keys(self, risk_control):
        """Should have expected keys."""
        summary = risk_control.summary()
        expected_keys = [
            "target_risk",
            "calibration_method",
            "risk_threshold",
            "n_calibration_samples",
        ]
        for key in expected_keys:
            assert key in summary


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 200
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
            "y": 3 * np.random.randn(n) + 5,
        }
    )


@pytest.fixture
def sample_model(sample_data):
    """Create a fitted base model for testing."""
    base_model = GradientBoostingRegressor(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model = ConformalRegressor(base_model)
    model.fit(sample_data, target="y")
    return model


@pytest.fixture
def risk_control(sample_model):
    """Create a ConformalRiskControl instance for testing."""
    risk_fn = asymmetric_loss(overprediction_penalty=1.0, underprediction_penalty=2.0)
    return ConformalRiskControl(
        sample_model,
        risk_fn,
        target_risk=0.1,
        random_state=42,
    )
