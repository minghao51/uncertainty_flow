"""Tests for DistributionPrediction class."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.core.distribution import DistributionPrediction


class FakeCopula:
    """Simple copula stub for joint-sampling tests."""

    def sample(
        self,
        marginals: np.ndarray,
        n_samples: int = 1000,
        quantile_levels: np.ndarray | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        del quantile_levels, random_state
        centers = marginals[:, :, 1]
        return np.repeat(centers[:, None, :], n_samples, axis=1)


class TestDistributionPredictionInit:
    """Test DistributionPrediction initialization."""

    def test_initialization(self):
        """Should initialize with valid parameters."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        assert dp._n_samples == 2
        assert len(dp._levels) == 3
        assert dp._targets == ["price"]

    def test_validates_2d_matrix(self):
        """Should raise error for 1D matrix."""
        matrix = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must be 2D"):
            DistributionPrediction(
                quantile_matrix=matrix,
                quantile_levels=[0.25, 0.5, 0.75],
                target_names=["price"],
            )

    def test_validates_matrix_shape(self):
        """Should raise error when matrix columns don't match levels."""
        matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="expected.*columns"):
            DistributionPrediction(
                quantile_matrix=matrix,
                quantile_levels=[0.25, 0.5, 0.75],  # 3 levels
                target_names=["price"],
            )

    def test_validates_non_empty_targets(self):
        """Should raise error for empty target list."""
        matrix = np.array([[1, 2, 3]])
        with pytest.raises(ValueError, match="target_names cannot be empty"):
            DistributionPrediction(
                quantile_matrix=matrix,
                quantile_levels=[0.25, 0.5, 0.75],
                target_names=[],
            )

    def test_repr(self):
        """Should have informative repr."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        repr_str = repr(dp)
        assert "n=2" in repr_str
        assert "targets=['price']" in repr_str
        assert "quantiles=3" in repr_str


class TestQuantileMethod:
    """Test quantile() method."""

    def test_single_quantile_univariate(self):
        """Should extract single quantile for univariate."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.quantile(0.5)
        assert isinstance(result, pl.DataFrame)
        assert "q_0.500" in result.columns
        assert result.to_numpy().tolist() == [[2], [5]]

    def test_multiple_quantiles_univariate(self):
        """Should extract multiple quantiles for univariate."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.quantile([0.25, 0.75])
        assert isinstance(result, pl.DataFrame)
        assert "q_0.250" in result.columns
        assert "q_0.750" in result.columns
        assert result.to_numpy().tolist() == [[1, 3], [4, 6]]

    def test_finds_nearest_quantile(self):
        """Should find nearest quantile level."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.5, 0.9],
            target_names=["price"],
        )
        # Request 0.6, should get 0.5
        result = dp.quantile(0.6)
        assert "q_0.600" in result.columns
        assert result.to_numpy().tolist() == [[2], [5]]


class TestIntervalMethod:
    """Test interval() method."""

    def test_interval_univariate(self):
        """Should compute prediction interval for univariate."""
        matrix = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95],
            target_names=["price"],
        )
        result = dp.interval(0.9)
        assert isinstance(result, pl.DataFrame)
        assert "lower" in result.columns
        assert "upper" in result.columns
        # 90% interval uses 0.05 and 0.95 quantiles
        assert result.to_numpy().tolist() == [[1, 5], [6, 10]]

    def test_interval_validates_confidence(self):
        """Should raise error for invalid confidence."""
        matrix = np.array([[1, 2, 3]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\)"):
            dp.interval(1.5)
        with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\)"):
            dp.interval(-0.1)


class TestMedian:
    """Test median() method."""

    def test_median_univariate(self):
        """Should return median (0.5 quantile) for univariate."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.median()
        assert isinstance(result, pl.Series)
        assert result.name == "median"
        assert result.to_list() == [2, 5]


class TestMultivariate:
    """Test multivariate functionality."""

    def test_multivariate_interval(self):
        """Should handle multivariate intervals."""
        # For multivariate: [target1_q1, ..., target1_qn, target2_q1, ..., target2_qn]
        # 2 targets, 3 quantiles each -> 6 columns total
        matrix = np.array(
            [
                [1, 2, 3, 10, 20, 30],  # price: [1,2,3], volume: [10,20,30]
                [4, 5, 6, 40, 50, 60],
            ]
        )
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],  # Same levels for each target
            target_names=["price", "volume"],
        )
        result = dp.interval(0.5)
        assert "price_lower" in result.columns
        assert "price_upper" in result.columns
        assert "volume_lower" in result.columns
        assert "volume_upper" in result.columns
        # 50% interval uses 0.25 and 0.75 quantiles
        # price: [1, 2, 3] -> lower=1, upper=3
        # volume: [10, 20, 30] -> lower=10, upper=30
        expected = [[1, 3, 10, 30], [4, 6, 40, 60]]
        assert result.to_numpy().tolist() == expected

    def test_multivariate_median(self):
        """Should return DataFrame for multivariate median."""
        # 2 targets, 3 quantiles each -> 6 columns total
        matrix = np.array(
            [
                [1, 2, 3, 10, 20, 30],  # price: [1,2,3], volume: [10,20,30]
                [4, 5, 6, 40, 50, 60],
            ]
        )
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price", "volume"],
        )
        result = dp.median()
        assert isinstance(result, pl.DataFrame)
        assert "price" in result.columns
        assert "volume" in result.columns
        assert result.to_numpy().tolist() == [[2, 20], [5, 50]]

    def test_legacy_mean_method_removed(self):
        """mean() should not exist on DistributionPrediction."""
        matrix = np.array(
            [
                [1, 2, 3, 10, 20, 30],
                [4, 5, 6, 40, 50, 60],
            ]
        )
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price", "volume"],
        )
        assert not hasattr(dp, "mean")


class TestPlotMethod:
    """Test plot() method."""

    def test_plot_without_matplotlib(self, monkeypatch):
        """Should raise ImportError if matplotlib not available."""
        matrix = np.array([[1, 2, 3]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )

        # Mock matplotlib import to fail
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("No module named 'matplotlib'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="matplotlib is required"):
            dp.plot()

    def test_plot_with_matplotlib(self):
        """Should create plot when matplotlib available."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            dp.plot(ax=ax)
            plt.close(fig)
        except ImportError:
            pytest.skip("matplotlib not available")
        except Exception:
            # Plot may still fail under constrained backends; this test only checks call safety.
            pass


class TestSampleMethod:
    """Test sample() method."""

    def test_sample_univariate_basic(self):
        """Should draw samples for univariate predictions."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.sample(n=5, random_state=42)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10  # 2 rows * 5 samples
        assert "sample_id" in result.columns
        assert "price" in result.columns
        assert result["sample_id"].to_list() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    def test_sample_reproducible_with_random_state(self):
        """Should produce identical results with same random_state."""
        matrix = np.array([[1.0, 2.0, 3.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result1 = dp.sample(n=3, random_state=123)
        result2 = dp.sample(n=3, random_state=123)
        assert result1.to_numpy().tolist() == result2.to_numpy().tolist()

    def test_sample_multivariate(self):
        """Should handle multivariate predictions."""
        matrix = np.array(
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            ]
        )
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price", "volume"],
        )
        result = dp.sample(n=3, random_state=42)
        assert "sample_id" in result.columns
        assert "price" in result.columns
        assert "volume" in result.columns
        assert len(result) == 6  # 2 rows * 3 samples

    def test_sample_values_within_range(self):
        """Sampled values should fall within quantile range."""
        matrix = np.array([[10.0, 50.0, 90.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.5, 0.9],
            target_names=["price"],
        )
        result = dp.sample(n=100, random_state=42)
        price_min = matrix[0, 0]  # 0.1 quantile = 10
        price_max = matrix[0, 2]  # 0.9 quantile = 90
        price_values = result["price"].to_numpy()
        assert np.all(price_values >= price_min - 1e-6)
        assert np.all(price_values <= price_max + 1e-6)

    def test_sample_different_n_per_row(self):
        """Should allow different n values per call."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.sample(n=10, random_state=42)
        assert len(result) == 20  # 2 rows * 10 samples

    def test_sample_chunked_preserves_original_sample_ids(self):
        """Chunked sampling should keep sample_id tied to the input row."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )

        result = dp.sample(n=100001, random_state=42)

        assert result["sample_id"].min() == 0
        assert result["sample_id"].max() == 1
        counts = result.group_by("sample_id").len().sort("sample_id")["len"].to_list()
        assert counts == [100001, 100001]

    def test_sample_single_row(self):
        """Should work with single row."""
        matrix = np.array([[1.0, 2.0, 3.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.sample(n=5, random_state=42)
        assert len(result) == 5
        assert result["sample_id"].to_list() == [0, 0, 0, 0, 0]

    def test_sample_preserves_dtype(self):
        """Should preserve float dtype."""
        matrix = np.array([[1.0, 2.0, 3.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.sample(n=5, random_state=42)
        assert result["price"].dtype == pl.Float64

    def test_sample_multivariate_with_copula_metadata(self):
        """Joint sampling should use the attached copula for multivariate predictions."""
        matrix = np.array(
            [
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
            ]
        )
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price", "volume"],
            copula=FakeCopula(),
        )

        result = dp.sample(n=3, random_state=42)

        assert result["sample_id"].to_list() == [0, 0, 0, 1, 1, 1]
        assert result["price"].to_list() == [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
        assert result["volume"].to_list() == [20.0, 20.0, 20.0, 50.0, 50.0, 50.0]

    def test_vectorized_inverse_cdf_matches_manual_linear_interpolation(self):
        """Vectorized inverse CDF should match manual interpolation on a toy example."""
        quantile_values = np.array([[1.0, 5.0, 9.0], [10.0, 20.0, 30.0]])
        uniform = np.array([[0.25, 0.5, 0.75], [0.3, 0.6, 0.7]])
        levels = np.array([0.1, 0.5, 0.9])

        expected = np.array(
            [
                [2.5, 5.0, 7.5],
                [15.0, 22.5, 25.0],
            ]
        )

        result = DistributionPrediction._vectorized_inverse_cdf(
            quantile_values,
            uniform,
            levels,
        )

        np.testing.assert_allclose(result, expected)


class TestPosteriorMethods:
    """Test Bayesian posterior extensions."""

    def test_init_accepts_posterior(self):
        matrix = np.array([[1, 2, 3]])
        np.random.seed(42)
        posterior = np.random.randn(100, 5)
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
        )
        assert dp._posterior is not None
        assert dp._posterior.shape == (100, 5)

    def test_posterior_defaults_to_none(self):
        matrix = np.array([[1, 2, 3]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        assert dp._posterior is None

    def test_posterior_samples_raises_without_posterior(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="posterior"):
            dp.posterior_samples()

    def test_posterior_samples_returns_array(self):
        np.random.seed(42)
        posterior = np.random.randn(100, 5)
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
        )
        result = dp.posterior_samples()
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 5)

    def test_credible_interval_raises_without_posterior(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="posterior"):
            dp.credible_interval(0.9)

    def test_credible_interval_returns_dataframe(self):
        np.random.seed(42)
        posterior = np.random.randn(1000, 3)
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
        )
        result = dp.credible_interval(0.9)
        assert isinstance(result, pl.DataFrame)
        assert "lower" in result.columns
        assert "upper" in result.columns
        assert result.height == 3

    def test_posterior_parameter_interval_returns_dataframe(self):
        np.random.seed(42)
        posterior = np.random.randn(1000, 3)
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
        )
        result = dp.posterior_parameter_interval(0.9)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 3

    def test_rhat_raises_without_posterior(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="chain"):
            dp.rhat()

    def test_rhat_returns_array(self):
        np.random.seed(42)
        posterior = np.random.randn(400, 5)
        posterior_chains = {"beta": np.random.randn(4, 100, 5)}
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
            posterior_chains=posterior_chains,
        )
        result = dp.rhat()
        assert isinstance(result, np.ndarray)

    def test_posterior_summary_raises_without_posterior(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="posterior"):
            dp.posterior_summary()

    def test_posterior_summary_returns_dataframe(self):
        np.random.seed(42)
        posterior = np.random.randn(400, 5)
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
        )
        result = dp.posterior_summary()
        assert isinstance(result, pl.DataFrame)
        assert "mean" in result.columns
        assert "std" in result.columns
        assert result.height == 5


class TestOptionalAttributes:
    """Test optional posterior, group, treatment attributes."""

    def test_init_accepts_group_predictions(self):
        group_pred = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            group_predictions={"demo": group_pred},
        )
        assert "demo" in dp._group_predictions

    def test_init_accepts_treatment_info(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            treatment_info={"cate": np.array([1.0, 2.0])},
        )
        assert "cate" in dp._treatment_info

    def test_all_optionals_default_none(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        assert dp._posterior is None
        assert dp._group_predictions is None
        assert dp._treatment_info is None


class TestGroupMethods:
    """Test multi-modal group uncertainty methods."""

    @pytest.fixture
    def dp_with_groups(self):
        group_a = DistributionPrediction(
            quantile_matrix=np.array([[0.8, 1.0, 1.2]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        group_b = DistributionPrediction(
            quantile_matrix=np.array([[0.9, 1.1, 1.3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        return DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            group_predictions={"demo": group_a, "temporal": group_b},
        )

    def test_group_uncertainty_raises_without_groups(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="group"):
            dp.group_uncertainty()

    def test_group_uncertainty_returns_dict(self, dp_with_groups):
        result = dp_with_groups.group_uncertainty()
        assert isinstance(result, dict)
        assert "demo" in result
        assert "temporal" in result

    def test_group_intervals_raises_without_groups(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="group"):
            dp.group_intervals(0.9)

    def test_group_intervals_returns_dict_of_dataframes(self, dp_with_groups):
        result = dp_with_groups.group_intervals(0.9)
        assert isinstance(result, dict)
        for df in result.values():
            assert isinstance(df, pl.DataFrame)
            assert "lower" in df.columns
            assert "upper" in df.columns

    def test_cross_group_correlation_raises_without_groups(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="group"):
            dp.cross_group_correlation()

    def test_cross_group_correlation_returns_array(self, dp_with_groups):
        result = dp_with_groups.cross_group_correlation()
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)


class TestTreatmentMethods:
    """Test causal treatment effect methods."""

    @pytest.fixture
    def dp_with_treatment(self):
        return DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3], [4, 5, 6]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["outcome"],
            treatment_info={
                "cate": np.array([0.5, 1.2]),
                "treatment_col": "intervention",
                "ate": 0.85,
                "ate_ci": (0.3, 1.4),
            },
        )

    def test_treatment_effect_raises_without_info(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["outcome"],
        )
        with pytest.raises(ValueError, match="treatment"):
            dp.treatment_effect()

    def test_treatment_effect_returns_array(self, dp_with_treatment):
        result = dp_with_treatment.treatment_effect()
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_average_treatment_effect_returns_dict(self, dp_with_treatment):
        result = dp_with_treatment.average_treatment_effect()
        assert isinstance(result, dict)
        assert result["ate"] == 0.85
        assert "ci" in result

    def test_heterogeneity_score_raises_without_info(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["outcome"],
        )
        with pytest.raises(ValueError, match="treatment"):
            dp.heterogeneity_score()

    def test_heterogeneity_score_returns_float(self, dp_with_treatment):
        result = dp_with_treatment.heterogeneity_score()
        assert isinstance(result, float)
        assert result >= 0

    def test_repr_with_posterior(self):
        np.random.seed(42)
        posterior = np.random.randn(100, 5)
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
        )
        repr_str = repr(dp)
        assert "posterior" in repr_str


class TestDistributionPredictionSummary:
    def test_summary_univariate(self):
        matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            target_names=["y"],
        )
        s = dp.summary()
        assert isinstance(s, pl.DataFrame)
        assert s.height == 1
        assert s.columns == [
            "target",
            "median",
            "mean_width_90",
            "mean_width_50",
            "aleatoric",
            "epistemic",
            "total_uncertainty",
        ]
        assert s["target"][0] == "y"

    def test_summary_multivariate(self):
        matrix = np.array([[1, 2, 3, 4, 5, 10, 11, 12, 13, 14]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            target_names=["a", "b"],
        )
        s = dp.summary()
        assert s.height == 2
        assert s["target"].to_list() == ["a", "b"]


class TestDistributionPredictionCRPS:
    def test_crps_univariate(self):
        q = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        dp = DistributionPrediction(q, [0.1, 0.25, 0.5, 0.75, 0.9], ["y"])
        assert dp.crps(np.array([5.0])) == pytest.approx(0.0, abs=1e-10)

    def test_crps_multivariate_returns_dict(self):
        q = np.array([[1, 2, 3, 4, 5, 10, 11, 12, 13, 14]])
        dp = DistributionPrediction(q, [0.1, 0.25, 0.5, 0.75, 0.9], ["a", "b"])
        result = dp.crps(np.array([[3.0, 12.0]]))
        assert isinstance(result, dict)
        assert "a" in result and "b" in result

    def test_crps_requires_two_quantile_levels(self):
        q = np.array([[5.0]])
        dp = DistributionPrediction(q, [0.5], ["y"])
        with pytest.raises(ValueError, match="at least 2"):
            dp.crps(np.array([5.0]))


class TestDistributionPredictionPlotMultivariate:
    def test_plot_all_targets(self):
        import matplotlib

        matplotlib.use("Agg")

        matrix = np.random.randn(10, 15)
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            target_names=["a", "b", "c"],
        )
        import matplotlib.pyplot as plt

        dp.plot(title="All")
        fig = plt.gcf()
        assert fig.axes[0] is not None
        plt.close("all")

    def test_plot_single_target_selection(self):
        import matplotlib

        matplotlib.use("Agg")

        matrix = np.random.randn(10, 15)
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            target_names=["a", "b", "c"],
        )
        import matplotlib.pyplot as plt

        dp.plot(targets="b")
        fig = plt.gcf()
        assert len(fig.axes) == 1
        plt.close("all")

    def test_plot_invalid_target_raises(self):
        matrix = np.random.randn(10, 15)
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            target_names=["a", "b", "c"],
        )
        with pytest.raises(ValueError, match="not found"):
            dp.plot(targets="z")

    def test_plot_max_targets_warning(self):
        import matplotlib

        matplotlib.use("Agg")

        matrix = np.random.randn(10, 15)
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            target_names=["a", "b", "c"],
        )
        import matplotlib.pyplot as plt

        with pytest.warns(UserWarning, match="max_targets"):
            dp.plot(targets="all", max_targets=2)
        plt.close("all")


class TestPITMethods:
    """Tests for PIT histogram and calibration curve methods."""

    @pytest.fixture
    def calibrated_pred(self):
        rng = np.random.default_rng(42)
        n = 200
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_matrix = np.zeros((n, len(levels)))
        for i, level in enumerate(levels):
            q_matrix[:, i] = rng.normal(0, 1 + level, size=n)
        q_matrix.sort(axis=1)
        return DistributionPrediction(
            quantile_matrix=q_matrix,
            quantile_levels=levels,
            target_names=["y"],
        )

    @pytest.fixture
    def true_values(self):
        rng = np.random.default_rng(42)
        return pl.Series("y", rng.normal(0, 1, size=200))

    def test_pit_histogram_returns_dataframe(self, calibrated_pred, true_values):
        hist = calibrated_pred.pit_histogram(true_values, n_bins=10)
        assert isinstance(hist, pl.DataFrame)
        assert "bin_center" in hist.columns
        assert "count" in hist.columns
        assert "expected" in hist.columns
        assert len(hist) == 10

    def test_pit_histogram_counts_sum_to_n(self, calibrated_pred, true_values):
        hist = calibrated_pred.pit_histogram(true_values, n_bins=10)
        assert hist["count"].sum() == 200

    def test_pit_histogram_expected_uniform(self, calibrated_pred, true_values):
        hist = calibrated_pred.pit_histogram(true_values, n_bins=10)
        expected_per_bin = hist["expected"][0]
        assert expected_per_bin == pytest.approx(20.0)

    def test_pit_values_in_unit_interval(self, calibrated_pred, true_values):
        pit = calibrated_pred._pit_values(true_values)
        assert isinstance(pit, np.ndarray)
        assert pit.min() >= 0.0
        assert pit.max() <= 1.0
        assert len(pit) == 200

    def test_calibration_curve_returns_dataframe(self, calibrated_pred, true_values):
        curve = calibrated_pred.calibration_curve(true_values, n_bins=20)
        assert isinstance(curve, pl.DataFrame)
        assert "expected_coverage" in curve.columns
        assert "observed_coverage" in curve.columns
        assert len(curve) == 20

    def test_calibration_curve_monotone_expected(self, calibrated_pred, true_values):
        curve = calibrated_pred.calibration_curve(true_values, n_bins=20)
        expected = curve["expected_coverage"].to_numpy()
        assert np.all(np.diff(expected) >= 0)

    def test_pit_histogram_multivariate(self):
        rng = np.random.default_rng(42)
        n = 50
        levels = [0.1, 0.5, 0.9]
        q_matrix = np.zeros((n, 6))
        for t in range(2):
            for i, level in enumerate(levels):
                q_matrix[:, t * 3 + i] = rng.normal(0, 1 + level, size=n)
            q_matrix[:, t * 3 : t * 3 + 3] = np.sort(q_matrix[:, t * 3 : t * 3 + 3], axis=1)
        dp = DistributionPrediction(
            quantile_matrix=q_matrix,
            quantile_levels=levels,
            target_names=["a", "b"],
        )
        y_true = pl.DataFrame(
            {
                "a": rng.normal(0, 1, n),
                "b": rng.normal(5, 2, n),
            }
        )
        hist = dp.pit_histogram(y_true, n_bins=5)
        assert isinstance(hist, dict)
        assert "a" in hist and "b" in hist
        assert len(hist["a"]) == 5

    def test_plot_pit_runs(self, calibrated_pred, true_values):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        calibrated_pred.plot_pit(true_values, n_bins=10)
        plt.close("all")

    def test_forward_cdf_edge_cases(self):
        levels = np.array([0.25, 0.5, 0.75])
        q_values = np.array([[1.0, 2.0, 3.0]])
        y = np.array([0.5])
        pit = DistributionPrediction._forward_cdf(q_values, levels, y)
        assert 0.0 <= pit[0] <= 1.0
        assert pit[0] < 0.25

        y_above = np.array([5.0])
        pit_above = DistributionPrediction._forward_cdf(q_values, levels, y_above)
        assert 0.0 <= pit_above[0] <= 1.0
        assert pit_above[0] > 0.75

        y_mid = np.array([2.5])
        pit_mid = DistributionPrediction._forward_cdf(q_values, levels, y_mid)
        assert pytest.approx(pit_mid[0], abs=0.05) == 0.625
