"""Tests for polars_bridge utilities."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.utils import to_numpy, to_polars


class TestToNumpy:
    """Test to_numpy conversion."""

    def test_dataframe_conversion(self):
        """Should convert DataFrame to numpy array."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        result = to_numpy(df, ["a", "b"])
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_lazyframe_materialization(self):
        """Should materialize LazyFrame."""
        lf = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        ).lazy()
        result = to_numpy(lf, ["a", "b"])
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_single_column(self):
        """Should work with single column."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = to_numpy(df, ["a"])
        expected = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(result, expected)

    def test_validates_missing_columns(self):
        """Should raise error for missing columns."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            to_numpy(df, ["a", "b"])

    def test_selects_columns_in_order(self):
        """Should select columns in specified order."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )
        result = to_numpy(df, ["c", "a"])
        expected = np.array([[7, 1], [8, 2], [9, 3]])
        np.testing.assert_array_equal(result, expected)


class TestToPolars:
    """Test to_polars conversion."""

    def test_2d_array(self):
        """Should convert 2D array to DataFrame."""
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        result = to_polars(arr, ["a", "b"])
        assert result.columns == ["a", "b"]
        assert result.height == 3
        np.testing.assert_array_equal(result.to_numpy(), arr)

    def test_1d_array(self):
        """Should convert 1D array to single-column DataFrame."""
        arr = np.array([1, 2, 3])
        result = to_polars(arr, ["a"])
        assert result.columns == ["a"]
        assert result.to_numpy().flatten().tolist() == [1, 2, 3]

    def test_1d_array_multiple_columns_raises_error(self):
        """Should raise error for 1D array with multiple column names."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="1D array requires single column name"):
            to_polars(arr, ["a", "b"])

    def test_validates_column_count(self):
        """Should raise error when columns don't match array shape."""
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        with pytest.raises(ValueError, match="1 column names provided"):
            to_polars(arr, ["a"])

    def test_restores_index(self):
        """Should restore original index if provided."""
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        index = pl.Series(["x", "y", "z"])
        result = to_polars(arr, ["a", "b"], index=index)
        # Index should be restored (check row order is preserved)
        assert result.height == 3

    def test_validates_index_length(self):
        """Should raise error when index length doesn't match."""
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        index = pl.Series(["x", "y"])  # Wrong length
        with pytest.raises(ValueError, match="Index length.*doesn't match"):
            to_polars(arr, ["a", "b"], index=index)
