"""Tests for CrossModalAggregator."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.multimodal import CrossModalAggregator
from uncertainty_flow.wrappers import ConformalRegressor

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "demographics": ["age", "income"],
    "temporal": ["lag_1", "lag_7"],
    "weather": ["temperature", "humidity"],
}


def _make_base_model() -> ConformalRegressor:
    return ConformalRegressor(
        base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
        random_state=42,
    )


@pytest.fixture
def multimodal_data():
    """100 samples with all feature group columns + target 'demand'."""
    rng = np.random.default_rng(42)
    n = 100
    return pl.DataFrame(
        {
            # demographics
            "age": rng.uniform(18, 80, n),
            "income": rng.uniform(20_000, 120_000, n),
            # temporal
            "lag_1": rng.normal(50, 10, n),
            "lag_7": rng.normal(45, 12, n),
            # weather
            "temperature": rng.normal(15, 8, n),
            "humidity": rng.uniform(30, 90, n),
            # target
            "demand": rng.normal(100, 20, n),
        }
    )


# ===================================================================
# Init
# ===================================================================


class TestCrossModalAggregatorInit:
    """Test CrossModalAggregator initialization."""

    def test_init_defaults(self):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS)
        assert agg.aggregation == "product"
        assert agg.random_state is None
        assert agg._fitted is False
        assert agg._group_models == {}

    def test_init_custom_aggregation(self):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, aggregation="independent")
        assert agg.aggregation == "independent"

    def test_init_copula_aggregation(self):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, aggregation="copula")
        assert agg.aggregation == "copula"

    def test_init_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="Invalid aggregation"):
            CrossModalAggregator(feature_groups=FEATURE_GROUPS, aggregation="unknown")

    def test_init_empty_groups_raises(self):
        with pytest.raises(ValueError, match="feature_groups cannot be empty"):
            CrossModalAggregator(feature_groups={})

    def test_init_random_state(self):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=123)
        assert agg.random_state == 123


# ===================================================================
# Fit
# ===================================================================


class TestCrossModalAggregatorFit:
    """Test CrossModalAggregator fit method."""

    def test_fit_returns_self(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        result = agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        assert result is agg

    def test_fit_sets_fitted_flag(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        assert not agg._fitted
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        assert agg._fitted is True

    def test_fit_creates_per_group_models(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        assert len(agg._group_models) == 3
        for group_name in FEATURE_GROUPS:
            assert group_name in agg._group_models

    def test_fit_stores_quantile_levels(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        assert agg._quantile_levels is not None
        assert len(agg._quantile_levels) == 11  # DEFAULT_QUANTILES

    def test_fit_requires_base_model(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        with pytest.raises(ValueError, match="base_model is required"):
            agg.fit(multimodal_data, target="demand")

    def test_fit_requires_target(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        with pytest.raises(ValueError, match="target is required"):
            agg.fit(multimodal_data, base_model=_make_base_model())

    def test_fit_with_lazyframe(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(
            multimodal_data.lazy(),
            target="demand",
            base_model=_make_base_model(),
        )
        assert agg._fitted is True


# ===================================================================
# Predict
# ===================================================================


class TestCrossModalAggregatorPredict:
    """Test CrossModalAggregator predict method."""

    def test_predict_before_fit_raises(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        with pytest.raises(ValueError, match="not fitted"):
            agg.predict(multimodal_data)

    def test_predict_returns_distribution_prediction(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        pred = agg.predict(multimodal_data)
        assert isinstance(pred, DistributionPrediction)

    def test_predict_correct_shape(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        pred = agg.predict(multimodal_data)
        assert pred._quantiles.shape == (100, 11)

    def test_predict_has_group_predictions(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        pred = agg.predict(multimodal_data)
        assert pred._group_predictions is not None
        assert len(pred._group_predictions) == 3
        for name in FEATURE_GROUPS:
            assert name in pred._group_predictions
            assert isinstance(pred._group_predictions[name], DistributionPrediction)

    def test_predict_with_lazyframe(self, multimodal_data):
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        pred = agg.predict(multimodal_data.lazy())
        assert isinstance(pred, DistributionPrediction)

    def test_predict_quantiles_sorted_per_row(self, multimodal_data):
        """Aggregated quantiles should be non-crossing per row."""
        agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        pred = agg.predict(multimodal_data)
        for i in range(pred._quantiles.shape[0]):
            row = pred._quantiles[i, :]
            assert list(row) == sorted(row), "Quantiles should be non-crossing"


# ===================================================================
# Integration / group methods
# ===================================================================


class TestGroupMethodsIntegration:
    """Test group-level analysis methods on aggregated predictions."""

    @pytest.fixture(autouse=True)
    def _fit_model(self, multimodal_data):
        self.agg = CrossModalAggregator(feature_groups=FEATURE_GROUPS, random_state=42)
        self.agg.fit(multimodal_data, target="demand", base_model=_make_base_model())
        self.pred = self.agg.predict(multimodal_data)

    def test_group_uncertainty(self):
        result = self.pred.group_uncertainty()
        assert isinstance(result, dict)
        assert set(result.keys()) == set(FEATURE_GROUPS.keys())
        for val in result.values():
            assert isinstance(val, float)
            assert val > 0

    def test_group_intervals(self):
        intervals = self.pred.group_intervals(confidence=0.9)
        assert isinstance(intervals, dict)
        assert set(intervals.keys()) == set(FEATURE_GROUPS.keys())
        for df in intervals.values():
            assert isinstance(df, pl.DataFrame)
            assert "lower" in df.columns
            assert "upper" in df.columns

    def test_cross_group_correlation(self):
        corr = self.pred.cross_group_correlation()
        assert isinstance(corr, np.ndarray)
        n_groups = len(FEATURE_GROUPS)
        assert corr.shape == (n_groups, n_groups)
        # Diagonal should be 1.0
        for i in range(n_groups):
            assert abs(corr[i, i] - 1.0) < 1e-10

    def test_independent_aggregation(self, multimodal_data):
        agg_ind = CrossModalAggregator(
            feature_groups=FEATURE_GROUPS, aggregation="independent", random_state=42
        )
        agg_ind.fit(multimodal_data, target="demand", base_model=_make_base_model())
        pred_ind = agg_ind.predict(multimodal_data)
        assert isinstance(pred_ind, DistributionPrediction)
        assert pred_ind._quantiles.shape == (100, 11)

    def test_copula_aggregation_falls_back(self, multimodal_data):
        """Copula aggregation currently falls back to independent."""
        agg_cop = CrossModalAggregator(
            feature_groups=FEATURE_GROUPS, aggregation="copula", random_state=42
        )
        agg_cop.fit(multimodal_data, target="demand", base_model=_make_base_model())
        pred_cop = agg_cop.predict(multimodal_data)
        assert isinstance(pred_cop, DistributionPrediction)
        assert pred_cop._quantiles.shape == (100, 11)
