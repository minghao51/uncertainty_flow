"""Tests for CausalUncertaintyEstimator."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.causal import CausalUncertaintyEstimator
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.wrappers import ConformalRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def causal_data():
    """Create synthetic causal inference data with known ATE=2.0.

    DGP: Y = 2*T + x1 + eps, where eps ~ N(0, 0.5).
    Treatment is assigned randomly (unconfounded).
    """
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    t = rng.binomial(1, 0.5, n)
    y = 2.0 * t + x1 + 0.5 * x2 + rng.standard_normal(n) * 0.5

    return pl.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "treatment": t,
            "outcome": y,
        }
    )


def _make_outcome_model(random_state=42):
    """Create a small ConformalRegressor for testing."""
    base = GradientBoostingRegressor(n_estimators=10, random_state=random_state, max_depth=3)
    return ConformalRegressor(base_model=base, random_state=random_state, auto_tune=False)


# ---------------------------------------------------------------------------
# TestCausalUncertaintyEstimatorInit
# ---------------------------------------------------------------------------


class TestCausalUncertaintyEstimatorInit:
    """Test initialization."""

    def test_init_defaults(self):
        """Should initialise with default parameters."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model())
        assert model.method == "doubly_robust"
        assert model.treatment_col == "treatment"
        assert model.propensity_model is None
        assert model.random_state is None
        assert model._fitted is False

    def test_init_custom_params(self):
        """Should accept custom parameters."""
        from sklearn.linear_model import LogisticRegression

        prop = LogisticRegression()
        model = CausalUncertaintyEstimator(
            outcome_model=_make_outcome_model(),
            propensity_model=prop,
            treatment_col="T",
            method="s_learner",
            random_state=123,
        )
        assert model.method == "s_learner"
        assert model.treatment_col == "T"
        assert model.propensity_model is prop
        assert model.random_state == 123

    def test_init_invalid_method_raises(self):
        """Should raise ValueError for invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            CausalUncertaintyEstimator(
                outcome_model=_make_outcome_model(),
                method="invalid",
            )


# ---------------------------------------------------------------------------
# TestCausalUncertaintyEstimatorFit
# ---------------------------------------------------------------------------


class TestCausalUncertaintyEstimatorFit:
    """Test fit method."""

    def test_fit_returns_self(self, causal_data):
        """Should return self for method chaining."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        result = model.fit(causal_data, target="outcome")
        assert result is model

    def test_fit_sets_fitted_flag(self, causal_data):
        """Should set _fitted flag to True."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        assert not model._fitted
        model.fit(causal_data, target="outcome")
        assert model._fitted is True

    def test_fit_stores_feature_cols(self, causal_data):
        """Should store feature column names (excludes target & treatment)."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        assert set(model._feature_cols_) == {"x1", "x2"}

    def test_fit_with_lazyframe(self, causal_data):
        """Should work with LazyFrame input."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data.lazy(), target="outcome")
        assert model._fitted is True


# ---------------------------------------------------------------------------
# TestCausalUncertaintyEstimatorPredict
# ---------------------------------------------------------------------------


class TestCausalUncertaintyEstimatorPredict:
    """Test predict method."""

    def test_predict_before_fit_raises(self):
        """Should raise error if model not fitted."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model())
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(pl.DataFrame({"x1": [1], "x2": [2], "treatment": [1], "outcome": [0]}))

    def test_predict_returns_distribution_prediction(self, causal_data):
        """Should return DistributionPrediction."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_output_shape(self, causal_data):
        """Should produce correct quantile matrix shape."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        # 200 samples, 3 quantile levels
        assert pred._quantiles.shape == (200, 3)

    def test_predict_quantile_levels(self, causal_data):
        """Should use [0.1, 0.5, 0.9] quantile levels."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        assert list(pred._levels) == [0.1, 0.5, 0.9]

    def test_predict_target_names(self, causal_data):
        """Should use 'treatment_effect' as target name."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        assert pred._targets == ["treatment_effect"]


# ---------------------------------------------------------------------------
# TestTreatmentEffectMethods
# ---------------------------------------------------------------------------


class TestTreatmentEffectMethods:
    """Test treatment effect outputs."""

    def test_treatment_effect_returns_array(self, causal_data):
        """treatment_effect() should return CATE array."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        cate = pred.treatment_effect()
        assert isinstance(cate, np.ndarray)
        assert len(cate) == 200

    def test_treatment_effect_near_true_ate(self, causal_data):
        """CATE mean should be near the true ATE of 2.0."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        cate_mean = float(np.mean(pred.treatment_effect()))
        assert abs(cate_mean - 2.0) < 1.0  # generous tolerance for small model

    def test_average_treatment_effect(self, causal_data):
        """average_treatment_effect() should return ATE with CI."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        ate_result = pred.average_treatment_effect()
        assert "ate" in ate_result
        assert "ci" in ate_result
        assert len(ate_result["ci"]) == 2
        assert ate_result["ci"][0] < ate_result["ci"][1]

    def test_ate_ci_contains_true_effect(self, causal_data):
        """ATE confidence interval should contain the true ATE of 2.0."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        ate_result = pred.average_treatment_effect()
        lo, hi = ate_result["ci"]
        assert lo <= 2.0 <= hi

    def test_heterogeneity_score(self, causal_data):
        """heterogeneity_score() should return a non-negative float."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        het = pred.heterogeneity_score()
        assert isinstance(het, float)
        assert het >= 0.0

    def test_treatment_info_keys(self, causal_data):
        """treatment_info should contain expected keys."""
        model = CausalUncertaintyEstimator(outcome_model=_make_outcome_model(), random_state=42)
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        info = pred._treatment_info
        assert "cate" in info
        assert "treatment_col" in info
        assert "ate" in info
        assert "ate_ci" in info
        assert info["treatment_col"] == "treatment"


# ---------------------------------------------------------------------------
# TestAlternativeMethods
# ---------------------------------------------------------------------------


class TestAlternativeMethods:
    """Test S-learner and T-learner methods."""

    def test_s_learner_fit_predict(self, causal_data):
        """S-learner should fit and predict successfully."""
        model = CausalUncertaintyEstimator(
            outcome_model=_make_outcome_model(),
            method="s_learner",
            random_state=42,
        )
        model.fit(causal_data, target="outcome")
        assert model._fitted is True

        pred = model.predict(causal_data)
        assert isinstance(pred, DistributionPrediction)
        assert pred._quantiles.shape == (200, 3)

        cate = pred.treatment_effect()
        assert len(cate) == 200

    def test_s_learner_ate_near_truth(self, causal_data):
        """S-learner ATE should be near the true ATE of 2.0."""
        model = CausalUncertaintyEstimator(
            outcome_model=_make_outcome_model(),
            method="s_learner",
            random_state=42,
        )
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        ate_result = pred.average_treatment_effect()
        assert abs(ate_result["ate"] - 2.0) < 1.5

    def test_t_learner_fit_predict(self, causal_data):
        """T-learner should fit and predict successfully."""
        model = CausalUncertaintyEstimator(
            outcome_model=_make_outcome_model(),
            method="t_learner",
            random_state=42,
        )
        model.fit(causal_data, target="outcome")
        assert model._fitted is True

        pred = model.predict(causal_data)
        assert isinstance(pred, DistributionPrediction)
        assert pred._quantiles.shape == (200, 3)

        cate = pred.treatment_effect()
        assert len(cate) == 200

    def test_t_learner_ate_near_truth(self, causal_data):
        """T-learner ATE should be near the true ATE of 2.0."""
        model = CausalUncertaintyEstimator(
            outcome_model=_make_outcome_model(),
            method="t_learner",
            random_state=42,
        )
        model.fit(causal_data, target="outcome")
        pred = model.predict(causal_data)
        ate_result = pred.average_treatment_effect()
        assert abs(ate_result["ate"] - 2.0) < 1.5
