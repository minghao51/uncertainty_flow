"""Tests for polars_bridge utilities."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.utils import to_numpy, to_numpy_series, to_polars
from uncertainty_flow.utils.polars_bridge import as_numpy


class TestToNumpy:
    """Test to_numpy conversion."""

    def test_dataframe_conversion(self):
        """Should convert DataFrame to numpy array."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = to_numpy(df, ["a", "b"])
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_lazyframe_materialization(self):
        """Should materialize LazyFrame."""
        lf = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
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
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = to_numpy(df, ["c", "a"])
        expected = np.array([[7, 1], [8, 2], [9, 3]])
        np.testing.assert_array_equal(result, expected)


class TestToPolars:
    """Test to_polars conversion."""

    def test_2d_array(self):
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        result = to_polars(arr, ["a", "b"])
        assert result.columns == ["a", "b"]
        assert result.height == 3
        np.testing.assert_array_equal(result.to_numpy(), arr)

    def test_1d_array(self):
        arr = np.array([1, 2, 3])
        result = to_polars(arr, ["a"])
        assert result.columns == ["a"]
        assert result.to_numpy().flatten().tolist() == [1, 2, 3]

    def test_1d_array_multiple_columns_raises_error(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="1D array requires single column name"):
            to_polars(arr, ["a", "b"])

    def test_validates_column_count(self):
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        with pytest.raises(ValueError, match="1 column names provided"):
            to_polars(arr, ["a"])

    def test_restores_index(self):
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        index = pl.Series(["x", "y", "z"])
        result = to_polars(arr, ["a", "b"], index=index)
        assert result.height == 3

    def test_validates_index_length(self):
        arr = np.array([[1, 4], [2, 5], [3, 6]])
        index = pl.Series(["x", "y"])
        with pytest.raises(ValueError, match="Index length.*doesn't match"):
            to_polars(arr, ["a", "b"], index=index)


class TestToNumpySeries:
    """Test to_numpy_series utility."""

    def test_series_conversion(self):
        s = pl.Series("a", [1, 2, 3])
        result = to_numpy_series(s)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_series_with_nulls(self):
        s = pl.Series("a", [1, None, 3])
        result = to_numpy_series(s)
        assert len(result) == 3

    def test_series_rejects_non_series(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match=r"Expected pl\.Series, got DataFrame\."):
            to_numpy_series(df)  # type: ignore[arg-type]


class TestAsNumpy:
    """Test as_numpy utility."""

    def test_as_numpy_converts_series_and_arrays(self):
        s = pl.Series("a", [1, 2, 3])
        arr = np.array([4.0, 5.0, 6.0])
        s_np, arr_np = as_numpy(s, arr)
        assert isinstance(s_np, np.ndarray)
        assert isinstance(arr_np, np.ndarray)
        np.testing.assert_array_equal(s_np, np.array([1, 2, 3]))
        np.testing.assert_array_equal(arr_np, np.array([4.0, 5.0, 6.0]))
