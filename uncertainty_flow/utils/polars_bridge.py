"""Polars bridge - conversion between Polars and NumPy."""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass

from .exceptions import error_invalid_data


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
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    # Validate all columns exist
    missing = [col for col in columns if col not in data.columns]
    if missing:
        error_invalid_data(f"Columns not found: {missing}")

    # Select columns
    data = data.select(columns)

    # Convert to numpy
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
