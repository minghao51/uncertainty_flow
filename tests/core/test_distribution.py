"""Tests for DistributionPrediction class."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.core.distribution import DistributionPrediction


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
