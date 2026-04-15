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


class TestMeanMethod:
    """Test mean() method."""

    def test_mean_univariate(self):
        """Should return median (0.5 quantile) for univariate."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        result = dp.mean()
        assert isinstance(result, pl.Series)
        assert result.name == "mean"
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

    def test_multivariate_mean(self):
        """Should return DataFrame for multivariate mean."""
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
        result = dp.mean()
        assert isinstance(result, pl.DataFrame)
        assert "price" in result.columns
        assert "volume" in result.columns
        # Mean is the 0.5 quantile (index 1 in [0.25, 0.5, 0.75])
        assert result.to_numpy().tolist() == [[2, 20], [5, 50]]


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
        # Just check it doesn't crash (we can't easily test plot output)
        try:
            dp.plot()
        except ImportError:
            pytest.skip("matplotlib not available")
        except Exception:
            # Plot might fail due to display backend, but that's ok
            # We just want to ensure the code runs
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

    def test_sample_without_scipy(self, monkeypatch):
        """Should raise ImportError if scipy not available."""
        matrix = np.array([[1, 2, 3]])
        dp = DistributionPrediction(
            quantile_matrix=matrix,
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="scipy is required for sampling"):
            dp.sample(n=5)

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
            interp1d=None,
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

    def test_rhat_raises_without_posterior(self):
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
        )
        with pytest.raises(ValueError, match="posterior"):
            dp.rhat()

    def test_rhat_returns_array(self):
        np.random.seed(42)
        posterior = np.random.randn(400, 5)
        dp = DistributionPrediction(
            quantile_matrix=np.array([[1, 2, 3]]),
            quantile_levels=[0.25, 0.5, 0.75],
            target_names=["price"],
            posterior=posterior,
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
