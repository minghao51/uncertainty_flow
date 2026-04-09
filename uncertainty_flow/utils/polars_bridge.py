"""Polars bridge - conversion between Polars and NumPy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass

from .exceptions import error_invalid_data


def materialize_lazyframe(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """
    Materialize LazyFrame if needed, return DataFrame as-is.

    Args:
        data: Polars DataFrame or LazyFrame

    Returns:
        Polars DataFrame (materialized if input was LazyFrame)

    Examples:
        >>> import polars as pl
        >>> lazy_df = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        >>> materialize_lazyframe(lazy_df)
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘
    """
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    return data  # type: ignore[return-value]


def to_numpy(
    data: pl.DataFrame | pl.LazyFrame,
    columns: list[str],
) -> np.ndarray:
    """
    Convert Polars DataFrame or LazyFrame to NumPy array.

    Args:
        data: Polars DataFrame or LazyFrame
        columns: List of column names to extract

    Returns:
        NumPy array with float64 dtype

    Raises:
        ValueError: If any column is missing from the data

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> to_numpy(df, ["a", "b"])
        array([[1., 4.],
               [2., 5.],
               [3., 6.]])
    """
    # Materialize LazyFrame if needed
    data = materialize_lazyframe(data)

    # Validate all columns exist
    missing = [col for col in columns if col not in data.columns]
    if missing:
        error_invalid_data(f"Columns not found: {missing}")

    # Select and convert in a single operation
    return data.select(columns).to_numpy()


def to_numpy_series_zero_copy(series: pl.Series) -> np.ndarray:
    """
    Convert Polars Series to NumPy array with zero-copy when possible.

    Falls back to regular conversion if zero-copy isn't possible
    (e.g., due to nulls or multiple chunks).

    Args:
        series: Polars Series

    Returns:
        NumPy array view of the series data (or copy if zero-copy not possible)

    Raises:
        ValueError: If input is not a pl.Series

    Examples:
        >>> import polars as pl
        >>> s = pl.Series("a", [1, 2, 3])
        >>> to_numpy_series_zero_copy(s)  # May be zero-copy
        array([1, 2, 3])
    """
    if not isinstance(series, pl.Series):
        error_invalid_data(
            f"Expected pl.Series, got {type(series).__name__}. "
            "Use DataFrame[column] to select a Series."
        )
    try:
        # Try zero-copy conversion (allow_copy=False is the modern API)
        return series.to_numpy(allow_copy=False)
    except (ValueError, RuntimeError):
        # Fall back to regular conversion if zero-copy not possible
        return series.to_numpy()


def to_numpy_zero_copy(
    data: pl.DataFrame | pl.LazyFrame,
    columns: list[str],
) -> np.ndarray:
    """
    Convert Polars DataFrame or LazyFrame columns to NumPy array.

    Note: For DataFrames, zero-copy is only possible when columns are
    contiguous and have compatible dtypes. This function attempts
    zero-copy but may create a copy when necessary.

    Args:
        data: Polars DataFrame or LazyFrame
        columns: List of column names to extract

    Returns:
        NumPy array (zero-copy when possible, otherwise a copy)

    Raises:
        ValueError: If any column is missing from the data

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> to_numpy_zero_copy(df, ["a", "b"])
        array([[1., 4.],
               [2., 5.],
               [3., 6.]])
    """
    # Materialize LazyFrame if needed
    data = materialize_lazyframe(data)

    # Validate all columns exist
    missing = [col for col in columns if col not in data.columns]
    if missing:
        error_invalid_data(f"Columns not found: {missing}")

    # Select columns and convert
    selected = data.select(columns)
    return selected.to_numpy()


def to_numpy_zero_copy_frame(data: pl.DataFrame | pl.LazyFrame) -> np.ndarray:
    """
    Convert entire Polars DataFrame or LazyFrame to NumPy array.

    Note: For DataFrames, zero-copy is only possible when columns are
    contiguous and have compatible dtypes. This function attempts
    zero-copy but may create a copy when necessary.

    Args:
        data: Polars DataFrame or LazyFrame

    Returns:
        NumPy array (zero-copy when possible, otherwise a copy)

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> to_numpy_zero_copy_frame(df)
        array([[1., 4.],
               [2., 5.],
               [3., 6.]])
    """
    # Materialize LazyFrame if needed
    data = materialize_lazyframe(data)

    return data.to_numpy()


def to_polars(
    array: np.ndarray,
    columns: list[str],
    index: pl.Series | None = None,
) -> pl.DataFrame:
    """
    Convert NumPy array back to Polars DataFrame.

    Args:
        array: NumPy array (1D or 2D)
        columns: Column names
        index: Optional row index to restore

    Returns:
        Polars DataFrame

    Raises:
        ValueError: If array shape doesn't match columns length

    Examples:
        >>> import numpy as np
        >>> import polars as pl
        >>> arr = np.array([[1, 4], [2, 5], [3, 6]])
        >>> to_polars(arr, ["a", "b"])
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
    """
    # Handle 1D arrays
    if array.ndim == 1:
        if len(columns) != 1:
            error_invalid_data(f"1D array requires single column name, got {len(columns)}")
        array = array.reshape(-1, 1)

    # Validate shape
    if array.shape[1] != len(columns):
        error_invalid_data(
            f"Array has {array.shape[1]} columns but {len(columns)} column names provided"
        )

    # Create DataFrame
    df = pl.DataFrame(array, schema=columns, orient="row")

    # Restore index if provided
    if index is not None:
        if len(index) != len(df):
            error_invalid_data(
                f"Index length {len(index)} doesn't match DataFrame length {len(df)}"
            )
        # Create mapping from row numbers to index values
        index_map = dict(enumerate(index.to_list()))
        df = (
            df.with_row_index("__index__")
            .with_columns(pl.col("__index__").replace_strict(index_map))
            .drop("__index__")
        )

    return df
