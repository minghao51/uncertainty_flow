"""Tests for polars_bridge utilities."""

import numpy as np
import polars as pl
import pytest

from uncertainty_flow.utils import to_numpy, to_polars
from uncertainty_flow.utils.polars_bridge import (
    to_numpy_series_zero_copy,
    to_numpy_zero_copy,
    to_numpy_zero_copy_frame,
)


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


class TestToNumpyZeroCopy:
    """Test zero-copy conversion utilities."""

    def test_zero_copy_single_col_remains_2d(self):
        """Single-column zero-copy conversions should preserve 2D shape."""
        df = pl.DataFrame({"single_col": [1, 2, 3]})

        result = to_numpy_zero_copy(df, ["single_col"])

        assert result.ndim == 2
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result, np.array([[1], [2], [3]]))

    def test_zero_copy_dataframe_single_column(self):
        """Should convert single column DataFrame with zero-copy when possible."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = to_numpy_zero_copy(df, ["a"])
        expected = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(result, expected)
        assert result.ndim == 2
        assert result.shape == (3, 1)

    def test_zero_copy_dataframe_multi_column(self):
        """Should convert multi-column DataFrame with zero-copy when possible."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        result = to_numpy_zero_copy(df, ["a", "b"])
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_zero_copy_lazyframe_materialization(self):
        """Should materialize LazyFrame before conversion."""
        lf = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        ).lazy()
        result = to_numpy_zero_copy(lf, ["a", "b"])
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_zero_copy_validates_missing_columns(self):
        """Should raise error for missing columns."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            to_numpy_zero_copy(df, ["a", "b"])

    def test_zero_copy_with_nulls_falls_back(self):
        """Should fall back to regular conversion when nulls present."""
        df = pl.DataFrame({"a": [1, None, 3]})
        result = to_numpy_zero_copy(df, ["a"])
        # Should still work, just with a copy
        assert result.shape == (3, 1)

    def test_zero_copy_series(self):
        """Should convert Series with zero-copy when possible."""
        s = pl.Series("a", [1, 2, 3])
        result = to_numpy_series_zero_copy(s)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_zero_copy_series_with_nulls_falls_back(self):
        """Should fall back to regular conversion when Series has nulls."""
        s = pl.Series("a", [1, None, 3])
        result = to_numpy_series_zero_copy(s)
        # Should still work, just with a copy
        assert len(result) == 3

    def test_zero_copy_series_rejects_non_series_input(self):
        """Should raise error for non-Series input."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(
            ValueError,
            match=r"Expected pl\.Series, got DataFrame\. Use DataFrame\[column\] to select a Series\.",
        ):
            to_numpy_series_zero_copy(df)  # type: ignore[arg-type]

    def test_zero_copy_frame(self):
        """Should convert entire DataFrame with zero-copy when possible."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        result = to_numpy_zero_copy_frame(df)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_zero_copy_frame_with_nulls_falls_back(self):
        """Should fall back to regular conversion when DataFrame has nulls."""
        df = pl.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = to_numpy_zero_copy_frame(df)
        # Should still work, just with a copy
        assert result.shape == (3, 2)

    def test_zero_copy_results_match_regular_conversion(self):
        """Zero-copy should produce identical results to regular conversion."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        result_zero_copy = to_numpy_zero_copy(df, ["a", "b"])
        result_regular = df.select(["a", "b"]).to_numpy()
        np.testing.assert_array_equal(result_zero_copy, result_regular)

    def test_zero_copy_series_results_match_regular_conversion(self):
        """Series zero-copy should produce identical results to regular conversion."""
        s = pl.Series("a", [1, 2, 3])
        result_zero_copy = to_numpy_series_zero_copy(s)
        result_regular = s.to_numpy()
        np.testing.assert_array_equal(result_zero_copy, result_regular)
