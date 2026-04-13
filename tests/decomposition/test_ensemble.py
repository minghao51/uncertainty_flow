"""Tests for refit-based EnsembleDecomposition."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.core import BaseUncertaintyModel, DistributionPrediction
from uncertainty_flow.decomposition import EnsembleDecomposition
from uncertainty_flow.models import QuantileForestForecaster


class LinearBootstrapToyModel(BaseUncertaintyModel):
    """Small bootstrap-sensitive model for decomposition tests."""

    def __init__(self):
        self.intercept_: float | None = None
        self.slope_: float | None = None
        self.width_: float | None = None

    def fit(self, data, target=None, **kwargs):
        if target is None:
            target = "y"

        x = data["x"].to_numpy()
        y = data[target].to_numpy()
        slope, intercept = np.polyfit(x, y, deg=1)
        residuals = y - (intercept + slope * x)

        self.intercept_ = float(intercept)
        self.slope_ = float(slope)
        self.width_ = float(max(np.std(residuals), 0.1))
        return self

    def predict(self, data):
        assert self.intercept_ is not None
        assert self.slope_ is not None
        assert self.width_ is not None

        x = data["x"].to_numpy()
        mean = self.intercept_ + self.slope_ * x
        quantile_matrix = np.column_stack(
            [
                mean - self.width_,
                mean,
                mean + self.width_,
            ]
        )
        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=[0.05, 0.5, 0.95],
            target_names=["y"],
        )


class TestEnsembleDecompositionInit:
    """Initialization and validation tests."""

    def test_init_with_required_params(self, sample_data):
        """Should accept a model factory and training data."""
        decomposer = EnsembleDecomposition(
            model_factory=lambda: QuantileForestForecaster(
                targets="y",
                horizon=1,
                n_estimators=10,
                min_samples_leaf=5,
                auto_tune=False,
            ),
            train_data=sample_data,
            n_bootstrap=3,
        )
        assert callable(decomposer.model_factory)
        assert decomposer.n_bootstrap == 3
        assert decomposer.train_data.height == sample_data.height

    def test_init_rejects_missing_factory(self, sample_data):
        """Should validate model_factory."""
        with pytest.raises(Exception):
            EnsembleDecomposition(model_factory=None, train_data=sample_data)  # type: ignore[arg-type]

    def test_init_rejects_missing_train_data(self):
        """Should validate train_data."""
        with pytest.raises(Exception):
            EnsembleDecomposition(
                model_factory=lambda: LinearBootstrapToyModel(),
                train_data=None,  # type: ignore[arg-type]
            )

    def test_summary_returns_refit_config(self, sample_data):
        """Should expose refit-based summary metadata."""
        decomposer = EnsembleDecomposition(
            model_factory=lambda: LinearBootstrapToyModel(),
            train_data=sample_data.select(["x", "y"]),
            target="y",
            random_state=123,
        )
        summary = decomposer.summary()
        assert summary["random_state"] == 123
        assert summary["refit_based"] is True


class TestEnsembleDecompositionRefitWorkflow:
    """Integration tests for refit-based decomposition."""

    def test_decompose_returns_expected_keys(self, sample_data):
        """Should return aleatoric, epistemic, and total."""
        decomposer = EnsembleDecomposition(
            model_factory=quantile_forest_factory,
            train_data=sample_data.select(["x1", "x2", "x3", "y"]),
            n_bootstrap=3,
            random_state=42,
        )
        result = decomposer.decompose(sample_data.select(["x1", "x2", "x3"]))
        assert set(result.keys()) == {"aleatoric", "epistemic", "total"}

    def test_decompose_total_equals_sum(self, sample_data):
        """Total should equal aleatoric plus epistemic."""
        decomposer = EnsembleDecomposition(
            model_factory=quantile_forest_factory,
            train_data=sample_data.select(["x1", "x2", "x3", "y"]),
            n_bootstrap=3,
            random_state=42,
        )
        result = decomposer.decompose(sample_data.select(["x1", "x2", "x3"]))
        assert result["total"] == pytest.approx(result["aleatoric"] + result["epistemic"])

    def test_decompose_by_sample_returns_epistemic_values(self, sample_data):
        """Per-sample decomposition should include non-zero epistemic values."""
        decomposer = EnsembleDecomposition(
            model_factory=lambda: LinearBootstrapToyModel(),
            train_data=sample_data.select(["x", "y"]),
            target="y",
            n_bootstrap=8,
            random_state=42,
        )
        result = decomposer.decompose_by_sample(pl.DataFrame({"x": [0.0, 3.0, 6.0]}))
        assert isinstance(result, pl.DataFrame)
        assert "epistemic" in result.columns
        assert (result["epistemic"] > 0).any()

    def test_decompose_is_deterministic_with_fixed_seed(self, sample_data):
        """Fixed random_state should produce stable results."""
        kwargs = {
            "model_factory": quantile_forest_factory,
            "train_data": sample_data.select(["x1", "x2", "x3", "y"]),
            "n_bootstrap": 3,
            "random_state": 7,
        }
        result_one = EnsembleDecomposition(**kwargs).decompose(
            sample_data.select(["x1", "x2", "x3"])
        )
        result_two = EnsembleDecomposition(**kwargs).decompose(
            sample_data.select(["x1", "x2", "x3"])
        )
        assert result_one == pytest.approx(result_two)

    def test_epistemic_grows_with_ensemble_disagreement(self, sample_data):
        """Farther evaluation points should show larger epistemic variance in the toy model."""
        decomposer = EnsembleDecomposition(
            model_factory=lambda: LinearBootstrapToyModel(),
            train_data=sample_data.select(["x", "y"]),
            target="y",
            n_bootstrap=12,
            random_state=123,
        )
        result = decomposer.decompose_by_sample(pl.DataFrame({"x": [0.0, 10.0]}))
        assert result["epistemic"][1] > result["epistemic"][0]

    def test_decompose_rejects_empty_eval_frame(self, sample_data):
        """Should reject empty evaluation frames."""
        decomposer = EnsembleDecomposition(
            model_factory=quantile_forest_factory,
            train_data=sample_data.select(["x1", "x2", "x3", "y"]),
        )
        with pytest.raises(Exception):
            decomposer.decompose(
                pl.DataFrame(schema={"x1": pl.Float64, "x2": pl.Float64, "x3": pl.Float64})
            )


class TestDistributionPredictionUncertaintyDecomposition:
    """Heuristic decomposition tests on DistributionPrediction."""

    def test_uncertainty_decomposition_returns_dict(self, fitted_sample_forecaster, sample_data):
        """Should stay available as a cheap prediction-level helper."""
        pred = fitted_sample_forecaster.predict(sample_data.select(["x1", "x2", "x3"]))
        result = pred.uncertainty_decomposition()
        assert isinstance(result, dict)

    def test_uncertainty_decomposition_respects_confidence(
        self, fitted_sample_forecaster, sample_data
    ):
        """Higher confidence should still produce larger heuristic uncertainty."""
        pred = fitted_sample_forecaster.predict(sample_data.select(["x1", "x2", "x3"]))
        result_90 = pred.uncertainty_decomposition(confidence=0.9)
        result_80 = pred.uncertainty_decomposition(confidence=0.8)
        assert result_90["total"] >= result_80["total"]


def quantile_forest_factory() -> QuantileForestForecaster:
    """Create a lightweight quantile forest for refit tests."""
    return QuantileForestForecaster(
        targets="y",
        horizon=1,
        n_estimators=10,
        min_samples_leaf=5,
        auto_tune=False,
    )


@pytest.fixture
def sample_data():
    """Create sample data for both native and toy decomposition tests."""
    rng = np.random.default_rng(42)
    n = 120
    x = np.linspace(0, 5, n)
    return pl.DataFrame(
        {
            "x": x,
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "y": 1.5 + 0.8 * x + rng.normal(scale=0.6 + 0.2 * x, size=n),
        }
    )


@pytest.fixture
def fitted_sample_forecaster(sample_data):
    """Create a fitted native forecaster for prediction-level heuristic tests."""
    model = quantile_forest_factory()
    model.fit(sample_data.select(["x1", "x2", "x3", "y"]))
    return model
