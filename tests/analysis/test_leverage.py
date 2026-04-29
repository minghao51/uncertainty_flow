"""Tests for FeatureLeverageAnalyzer."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.analysis import FeatureLeverageAnalyzer
from uncertainty_flow.models import QuantileForestForecaster


class TestFeatureLeverageAnalyzerInit:
    """Test FeatureLeverageAnalyzer initialization."""

    def test_init_with_model(self, sample_forecaster):
        """Should initialize with a fitted model."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        assert analyzer.model is sample_forecaster
        assert analyzer.confidence == 0.9
        assert analyzer.n_perturbations == 100
        assert analyzer.n_bins == 10

    def test_init_with_custom_params(self, sample_forecaster):
        """Should accept custom parameters."""
        analyzer = FeatureLeverageAnalyzer(
            sample_forecaster,
            confidence=0.8,
            n_perturbations=50,
            n_bins=5,
            leverage_threshold=0.3,
            random_state=42,
        )
        assert analyzer.confidence == 0.8
        assert analyzer.n_perturbations == 50
        assert analyzer.n_bins == 5
        assert analyzer.leverage_threshold == 0.3
        assert analyzer.random_state == 42

    def test_summary_returns_config(self, sample_forecaster):
        """Should return analyzer configuration."""
        analyzer = FeatureLeverageAnalyzer(
            sample_forecaster,
            confidence=0.95,
            random_state=123,
        )
        summary = analyzer.summary()
        assert summary["confidence"] == 0.95
        assert summary["random_state"] == 123
        assert summary["effective_prediction_row_budget"] == 800


class TestFeatureLeverageAnalyzerOutput:
    """Test FeatureLeverageAnalyzer output structure."""

    def test_analyze_returns_dataframe(self, sample_forecaster, sample_features):
        """Should return a Polars DataFrame."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        assert isinstance(result, pl.DataFrame)

    def test_analyze_has_expected_columns(self, sample_forecaster, sample_features):
        """Should have expected columns."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        expected_cols = [
            "feature",
            "aleatoric_score",
            "epistemic_score",
            "leverage_score",
            "recommendation",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_analyze_sorted_by_leverage(self, sample_forecaster, sample_features):
        """Should be sorted by leverage_score descending."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        if result.height > 1:
            scores = result["leverage_score"].to_list()
            assert scores == sorted(scores, reverse=True)


class TestFeatureLeverageAnalyzerLeverageScore:
    """Test leverage score computation."""

    def test_leverage_score_non_negative(self, sample_forecaster, sample_features):
        """Leverage scores should be non-negative."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        assert (result["leverage_score"] >= 0).all()

    def test_leverage_score_detects_important_features(self, sample_forecaster, sample_data):
        """Features that affect predictions more should have higher leverage."""
        # Create synthetic data where x1 clearly affects variance more
        np.random.seed(42)
        n = 200
        x1 = np.linspace(0, 10, n)
        x2 = np.random.randn(n)
        # y has variance that increases with x1 (heteroscedastic)
        y = 5 * x1 + 2 * x2 + x1 * np.random.randn(n)

        data = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
        sample_forecaster.fit(data, target="y")

        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(data.select(["x1", "x2"]))

        # x1 should have higher leverage since it affects variance
        if result.height == 2:
            x1_leverage = result.filter(pl.col("feature") == "x1")["leverage_score"][0]
            x2_leverage = result.filter(pl.col("feature") == "x2")["leverage_score"][0]
            assert x1_leverage > x2_leverage


class TestFeatureLeverageAnalyzerDecomposition:
    """Test aleatoric/epistemic decomposition."""

    def test_aleatoric_score_non_negative(self, sample_forecaster, sample_features):
        """Aleatoric scores should be non-negative."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        assert (result["aleatoric_score"] >= 0).all()

    def test_epistemic_score_non_negative(self, sample_forecaster, sample_features):
        """Epistemic scores should be non-negative."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        assert (result["epistemic_score"] >= 0).all()


class TestFeatureLeverageAnalyzerRecommendations:
    """Test recommendation generation."""

    def test_recommendation_is_string(self, sample_forecaster, sample_features):
        """Recommendations should be strings."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        assert result["recommendation"].dtype == pl.String

    def test_recommendation_has_known_values(self, sample_forecaster, sample_features):
        """Recommendations should be from known set."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        result = analyzer.analyze(sample_features)
        known_rec = {
            "Accept uncertainty (inherently noisy)",
            "Collect more training data",
            "High leverage - prioritize accurate measurement",
            "Low leverage - can approximate for efficiency",
        }
        for rec in result["recommendation"].to_list():
            assert rec in known_rec


class TestFeatureLeverageAnalyzerEdgeCases:
    """Test edge cases."""

    def test_analyze_with_empty_dataframe(self, sample_forecaster):
        """Should raise InvalidDataError for empty DataFrame."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        empty_df = pl.DataFrame()
        with pytest.raises(Exception):  # InvalidDataError
            analyzer.analyze(empty_df)

    def test_analyze_with_constant_feature(self, sample_forecaster):
        """Should skip constant features."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        np.random.seed(42)
        data = pl.DataFrame(
            {
                "x1": np.random.randn(50),
                "x2": np.random.randn(50),
                "x3": [5.0] * 50,  # constant feature
            }
        )
        result = analyzer.analyze(data)
        features = result["feature"].to_list()
        assert "x3" not in features  # constant feature should be skipped
        # At least one of x1 or x2 should be present
        assert len(features) >= 1

    def test_analyze_single_feature(self, sample_forecaster):
        """Should analyze with subset of features available."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        np.random.seed(42)
        # Note: The model was trained on x1, x2, x3
        # We need to provide all features, but we can test the analyzer still works
        data = pl.DataFrame(
            {
                "x1": np.random.randn(50),
                "x2": np.random.randn(50),
                "x3": np.random.randn(50),
            }
        )
        result = analyzer.analyze(data)
        # Should return results for all non-constant features
        assert result.height >= 1
        assert "feature" in result.columns

    def test_effective_perturbation_count_scales_with_frame_size(self, sample_forecaster):
        """Should use fewer repeats for larger evaluation frames."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster, n_perturbations=100)

        assert analyzer._effective_perturbation_count(50) == 16
        assert analyzer._effective_perturbation_count(100) == 8
        assert analyzer._effective_perturbation_count(500) == 1


class TestFeatureLeverageAnalyzerMultivariate:
    """Test multivariate extension."""

    def test_analyze_multivariate_method_exists(self, sample_forecaster):
        """Should have analyze_multivariate method."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        assert hasattr(analyzer, "analyze_multivariate")
        assert callable(analyzer.analyze_multivariate)

    def test_analyze_multivariate_falls_back_for_univariate(self, sample_forecaster):
        """Should fall back to standard analyze for univariate models."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        np.random.seed(42)
        data = pl.DataFrame(
            {
                "x1": np.random.randn(50),
                "x2": np.random.randn(50),
                "x3": np.random.randn(50),
            }
        )
        result = analyzer.analyze_multivariate(data)
        assert isinstance(result, pl.DataFrame)
        # Should have same columns as standard analyze
        assert "feature" in result.columns
        assert "leverage_score" in result.columns

    def test_analyze_multivariate_returns_target_specific_rows(
        self,
        multivariate_forecaster,
        multivariate_features,
    ):
        """Should return one leverage row per feature, including joint scope."""
        analyzer = FeatureLeverageAnalyzer(
            multivariate_forecaster,
            n_perturbations=4,
            random_state=42,
        )
        result = analyzer.analyze_multivariate(multivariate_features)

        assert "target" in result.columns
        assert "dependence_shift" in result.columns
        assert set(result["target"].to_list()) == {"y1", "y2", "joint"}
        assert result.height == len(multivariate_features.columns) * 3
        assert (result["leverage_score"] >= 0).all()
        assert (result["aleatoric_score"] >= 0).all()
        assert (result["epistemic_score"] >= 0).all()
        assert (result["dependence_shift"] >= 0).all()

    def test_analyze_multivariate_is_not_duplicated_across_targets(
        self,
        multivariate_forecaster,
        multivariate_features,
    ):
        """Different targets should yield distinct leverage values for at least one feature."""
        analyzer = FeatureLeverageAnalyzer(
            multivariate_forecaster,
            n_perturbations=4,
            random_state=42,
        )
        result = analyzer.analyze_multivariate(multivariate_features)

        paired = result.pivot(
            values="leverage_score",
            index="feature",
            on="target",
        )
        differences = np.abs(paired["y1"].to_numpy() - paired["y2"].to_numpy())
        assert np.any(differences > 1e-9)

    def test_analyze_multivariate_reports_joint_dependence_shift(
        self,
        multivariate_forecaster,
        multivariate_features,
    ):
        """Joint rows should expose a dependence-shift signal."""
        analyzer = FeatureLeverageAnalyzer(
            multivariate_forecaster,
            n_perturbations=4,
            random_state=42,
        )
        result = analyzer.analyze_multivariate(multivariate_features)

        joint_rows = result.filter(pl.col("target") == "joint")
        per_target_rows = result.filter(pl.col("target") != "joint")

        assert joint_rows.height == len(multivariate_features.columns)
        assert (per_target_rows["dependence_shift"] == 0.0).all()
        assert np.any(joint_rows["dependence_shift"].to_numpy() > 0)


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample training data."""
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
def sample_features():
    """Create sample feature data for analysis."""
    np.random.seed(123)
    n = 100
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "x3": np.random.randn(n),
        }
    )


@pytest.fixture
def sample_forecaster(sample_data):
    """Create a fitted forecaster for testing."""
    model = QuantileForestForecaster(
        targets="y",
        horizon=1,
        n_estimators=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(sample_data)
    return model


@pytest.fixture
def multivariate_data():
    """Create multi-target data with different uncertainty structure per target."""
    rng = np.random.default_rng(7)
    n = 160
    x1 = np.linspace(0, 6, n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y1 = 2.0 * x1 + 0.5 * x2 + rng.normal(scale=0.5 + 0.3 * x1, size=n)
    y2 = -1.5 * x1 + 1.8 * x3 + rng.normal(scale=1.2 + 0.1 * np.abs(x2), size=n)
    return pl.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y1": y1,
            "y2": y2,
        }
    )


@pytest.fixture
def multivariate_features(multivariate_data):
    """Feature frame for multivariate leverage tests."""
    return multivariate_data.select(["x1", "x2", "x3"])


@pytest.fixture
def multivariate_forecaster(multivariate_data):
    """Create a fitted multivariate forecaster."""
    model = QuantileForestForecaster(
        targets=["y1", "y2"],
        horizon=1,
        n_estimators=12,
        min_samples_leaf=5,
        auto_tune=False,
        random_state=42,
    )
    model.fit(multivariate_data)
    return model


@pytest.fixture
def sample_multivariate_data():
    """Create sample multivariate training data."""
    np.random.seed(42)
    n = 200
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "y1": 3 * np.random.randn(n) + 5,
            "y2": 2 * np.random.randn(n) + 3,
        }
    )


@pytest.fixture
def sample_multivariate_forecaster(sample_multivariate_data):
    """Create a fitted multivariate forecaster for testing."""
    from uncertainty_flow.wrappers import ConformalForecaster

    base_model = GradientBoostingRegressor(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model = ConformalForecaster(
        base_model=base_model,
        horizon=1,
        targets=["y1", "y2"],
    )
    model.fit(sample_multivariate_data)
    return model
