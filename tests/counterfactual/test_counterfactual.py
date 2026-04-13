"""Tests for UncertaintyExplainer and search strategies."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.counterfactual import UncertaintyExplainer
from uncertainty_flow.counterfactual.search import EvolutionarySearch, GradientSearch, SearchResult
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.wrappers import ConformalRegressor


class TestSearchResult:
    """Test SearchResult class."""

    def test_search_result_init(self):
        """Should initialize with all required attributes."""
        original = pl.DataFrame({"x1": [1.0], "x2": [2.0]})
        counterfactual = pl.DataFrame({"x1": [1.5], "x2": [2.0]})
        changes = {"x1": 0.5, "x2": 0.0}

        result = SearchResult(
            counterfactual=counterfactual,
            original=original,
            changes=changes,
            interval_width_reduction=0.5,
            original_width=10.0,
            new_width=5.0,
        )

        assert result.interval_width_reduction == 0.5
        assert result.original_width == 10.0
        assert result.new_width == 5.0
        assert result.changes == changes

    def test_search_result_to_polars(self):
        """Should convert to Polars DataFrame."""
        original = pl.DataFrame({"x1": [1.0], "x2": [2.0]})
        counterfactual = pl.DataFrame({"x1": [1.5], "x2": [2.0]})
        changes = {"x1": 0.5, "x2": 0.0}

        result = SearchResult(
            counterfactual=counterfactual,
            original=original,
            changes=changes,
            interval_width_reduction=0.5,
            original_width=10.0,
            new_width=5.0,
        )

        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "feature" in df.columns
        assert "change" in df.columns
        assert "width_reduction_pct" in df.columns
        assert df["width_reduction_pct"][0] == 50.0


class TestEvolutionarySearch:
    """Test EvolutionarySearch class."""

    def test_evolutionary_search_init(self, sample_forecaster):
        """Should initialize with default parameters."""
        searcher = EvolutionarySearch(sample_forecaster)
        assert searcher.model is sample_forecaster
        assert searcher.confidence == 0.9
        assert searcher.population_size == 50
        assert searcher.n_generations == 100

    def test_evolutionary_search_custom_params(self, sample_forecaster):
        """Should accept custom parameters."""
        searcher = EvolutionarySearch(
            sample_forecaster,
            confidence=0.8,
            population_size=30,
            n_generations=50,
            random_state=42,
        )
        assert searcher.confidence == 0.8
        assert searcher.population_size == 30
        assert searcher.n_generations == 50
        assert searcher.random_state == 42

    def test_evolutionary_search_returns_result(self, sample_forecaster, sample_single_row):
        """Should return SearchResult."""
        searcher = EvolutionarySearch(sample_forecaster, random_state=42)
        result = searcher.search(
            sample_single_row,
            target_reduction=0.3,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(result, SearchResult)
        assert hasattr(result, "counterfactual")
        assert hasattr(result, "changes")
        assert hasattr(result, "interval_width_reduction")

    def test_evolutionary_search_reduces_width(self, sample_forecaster, sample_single_row):
        """Should reduce interval width (or keep same)."""
        searcher = EvolutionarySearch(sample_forecaster, random_state=42)
        result = searcher.search(
            sample_single_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        # New width should be <= original width (allowing for some tolerance)
        assert result.new_width <= result.original_width * 1.1

    def test_evolutionary_search_respects_bounds(self, sample_forecaster, sample_single_row):
        """Should respect feature bounds."""
        bounds = {"x1": (0, 3), "x2": (0, 3), "x3": (0, 3)}
        searcher = EvolutionarySearch(sample_forecaster, random_state=42)
        result = searcher.search(
            sample_single_row,
            target_reduction=0.2,
            feature_bounds=bounds,
        )

        for col in bounds.keys():
            cf_val = result.counterfactual[col][0]
            lower, upper = bounds[col]
            assert lower <= cf_val <= upper

    def test_evolutionary_search_fixed_features(self, sample_forecaster, sample_single_row):
        """Should keep fixed features unchanged."""
        searcher = EvolutionarySearch(sample_forecaster, random_state=42)
        result = searcher.search(
            sample_single_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
            fixed_features=["x1"],
        )

        # x1 should remain unchanged
        orig_x1 = sample_single_row["x1"][0]
        cf_x1 = result.counterfactual["x1"][0]
        assert orig_x1 == cf_x1


class TestGradientSearch:
    """Test GradientSearch class."""

    def test_gradient_search_init(self, sample_forecaster):
        """Should initialize with default parameters."""
        searcher = GradientSearch(sample_forecaster)
        assert searcher.model is sample_forecaster
        assert searcher.confidence == 0.9
        assert searcher.learning_rate == 0.01
        assert searcher.n_iterations == 1000

    def test_gradient_search_custom_params(self, sample_forecaster):
        """Should accept custom parameters."""
        searcher = GradientSearch(
            sample_forecaster,
            confidence=0.8,
            learning_rate=0.05,
            n_iterations=500,
            random_state=42,
        )
        assert searcher.confidence == 0.8
        assert searcher.learning_rate == 0.05
        assert searcher.n_iterations == 500

    def test_gradient_search_returns_result(self, sample_forecaster, sample_single_row):
        """Should return SearchResult (uses finite difference fallback)."""
        searcher = GradientSearch(sample_forecaster, random_state=42)
        result = searcher.search(
            sample_single_row,
            target_reduction=0.3,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(result, SearchResult)
        assert hasattr(result, "counterfactual")
        assert hasattr(result, "changes")


class TestUncertaintyExplainer:
    """Test UncertaintyExplainer class."""

    def test_explainer_init(self, sample_forecaster):
        """Should initialize with default parameters."""
        explainer = UncertaintyExplainer(sample_forecaster)
        assert explainer.model is sample_forecaster
        assert explainer.confidence == 0.9
        assert explainer.method == "auto"

    def test_explainer_custom_params(self, sample_forecaster):
        """Should accept custom parameters."""
        explainer = UncertaintyExplainer(
            sample_forecaster,
            confidence=0.8,
            method="evolutionary",
            random_state=42,
        )
        assert explainer.confidence == 0.8
        assert explainer.method == "evolutionary"
        assert explainer.random_state == 42

    def test_explainer_invalid_method(self, sample_forecaster):
        """Should raise ValueError for invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            UncertaintyExplainer(sample_forecaster, method="invalid")

    def test_explainer_summary(self, sample_forecaster):
        """Should return configuration summary."""
        explainer = UncertaintyExplainer(sample_forecaster, confidence=0.95)
        summary = explainer.summary()
        assert summary["confidence"] == 0.95
        assert summary["method"] == "auto"
        assert "model_type" in summary


class TestUncertaintyExplainerExplain:
    """Test explain_uncertainty method."""

    def test_explain_uncertainty_returns_result(self, sample_forecaster, sample_single_row):
        """Should return SearchResult."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        result = explainer.explain_uncertainty(
            sample_single_row,
            target_reduction=0.3,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(result, SearchResult)

    def test_explain_uncertainty_with_target_reduction(self, sample_forecaster, sample_single_row):
        """Should attempt to achieve target reduction."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        result = explainer.explain_uncertainty(
            sample_single_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        # Check that some reduction was achieved or attempted
        assert result.interval_width_reduction >= 0

    def test_explain_uncertainty_respects_fixed_features(
        self, sample_forecaster, sample_single_row
    ):
        """Should keep fixed features unchanged."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        result = explainer.explain_uncertainty(
            sample_single_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
            fixed_features=["x1"],
        )

        orig_x1 = sample_single_row["x1"][0]
        cf_x1 = result.counterfactual["x1"][0]
        assert orig_x1 == cf_x1

    def test_explain_uncertainty_to_polars(self, sample_forecaster, sample_single_row):
        """Should convert result to Polars DataFrame."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        result = explainer.explain_uncertainty(
            sample_single_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )

        df = result.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "feature" in df.columns

    def test_explain_uncertainty_multiple_rows_raises(
        self, sample_forecaster, sample_features_small
    ):
        """Should direct multi-row inputs to explain_batch instead of truncating."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        with pytest.raises(Exception, match="exactly one row"):
            explainer.explain_uncertainty(
                sample_features_small.head(2),
                target_reduction=0.2,
                feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
            )


class TestUncertaintyExplainerBatch:
    """Test explain_batch method."""

    def test_explain_batch_returns_list(self, sample_forecaster, sample_features_small):
        """Should return list of SearchResults."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        results = explainer.explain_batch(
            sample_features_small.head(3),
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)


class TestUncertaintyExplainerCompareFeatures:
    """Test compare_features method."""

    def test_compare_features_returns_dataframe(self, sample_forecaster, sample_single_row):
        """Should return comparison DataFrame."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        comparison = explainer.compare_features(
            sample_single_row,
            features=["x1", "x2"],
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(comparison, pl.DataFrame)
        assert "feature" in comparison.columns
        assert "effectiveness" in comparison.columns

    def test_compare_features_sorts_by_effectiveness(self, sample_forecaster, sample_single_row):
        """Should sort by effectiveness descending."""
        explainer = UncertaintyExplainer(sample_forecaster, random_state=42)
        comparison = explainer.compare_features(
            sample_single_row,
            features=["x1", "x2", "x3"],
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        if comparison.height > 1:
            effectiveness = comparison["effectiveness"].to_list()
            assert effectiveness == sorted(effectiveness, reverse=True)


class TestUncertaintyExplainerEdgeCases:
    """Test edge cases."""

    def test_explain_empty_dataframe(self, sample_forecaster):
        """Should raise error for empty DataFrame."""

        explainer = UncertaintyExplainer(sample_forecaster)
        empty_df = pl.DataFrame()
        with pytest.raises(Exception):  # InvalidDataError
            explainer.explain_uncertainty(empty_df, target_reduction=0.3)

    def test_explain_with_conformal_regressor(self, sample_data):
        """Should work with ConformalRegressor."""
        base_model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model = ConformalRegressor(base_model=base_model, random_state=42)
        model.fit(sample_data, target="y")

        explainer = UncertaintyExplainer(model, random_state=42)
        test_row = sample_data.head(1).drop("y")
        result = explainer.explain_uncertainty(
            test_row,
            target_reduction=0.2,
            feature_bounds={"x1": (-5, 5), "x2": (-5, 5), "x3": (-5, 5)},
        )
        assert isinstance(result, SearchResult)


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
def sample_single_row():
    """Create single row for testing."""
    np.random.seed(123)
    return pl.DataFrame(
        {
            "x1": [0.5],
            "x2": [-0.3],
            "x3": [1.2],
        }
    )


@pytest.fixture
def sample_features_small():
    """Create small feature dataset for batch testing."""
    np.random.seed(123)
    n = 5
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
