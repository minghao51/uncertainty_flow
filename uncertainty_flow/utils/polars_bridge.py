"""Polars bridge - conversion between Polars and NumPy."""

from __future__ import annotations

import numpy as np
import polars as pl

from .exceptions import InvalidDataError


def materialize_lazyframe(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Materialize LazyFrame if needed, return DataFrame as-is."""
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    return data  # type: ignore[return-value]


def to_numpy(
    data: pl.DataFrame | pl.LazyFrame,
    columns: list[str],
) -> np.ndarray:
    """Convert Polars DataFrame or LazyFrame to NumPy array.

    Raises:
        InvalidDataError: If any column is missing from the data.
    """
    data = materialize_lazyframe(data)

    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise InvalidDataError(f"Columns not found: {missing}")

    return data.select(columns).to_numpy()


def to_numpy_series(series: pl.Series) -> np.ndarray:
    """Convert Polars Series to NumPy array, zero-copy when possible.

    Falls back to regular conversion if zero-copy isn't possible.

    Raises:
        InvalidDataError: If input is not a pl.Series.
    """
    if not isinstance(series, pl.Series):
        raise InvalidDataError(
            f"Expected pl.Series, got {type(series).__name__}. "
            "Use DataFrame[column] to select a Series."
        )
    try:
        return series.to_numpy(allow_copy=False)
    except (ValueError, RuntimeError):
        return series.to_numpy()


def to_polars(
    array: np.ndarray,
    columns: list[str],
    index: pl.Series | None = None,
) -> pl.DataFrame:
    """Convert NumPy array back to Polars DataFrame.

    Raises:
        InvalidDataError: If array shape doesn't match columns length.
    """
    if array.ndim == 1:
        if len(columns) != 1:
            raise InvalidDataError(f"1D array requires single column name, got {len(columns)}")
        array = array.reshape(-1, 1)

    if array.shape[1] != len(columns):
        raise InvalidDataError(
            f"Array has {array.shape[1]} columns but {len(columns)} column names provided"
        )

    df = pl.DataFrame(array, schema=columns, orient="row")

    if index is not None:
        if len(index) != len(df):
            raise InvalidDataError(
                f"Index length {len(index)} doesn't match DataFrame length {len(df)}"
            )
        index_map = dict(enumerate(index.to_list()))
        df = (
            df.with_row_index("__index__")
            .with_columns(pl.col("__index__").replace_strict(index_map))
            .drop("__index__")
        )

    return df


def as_numpy(*arrays: pl.Series | np.ndarray) -> tuple[np.ndarray, ...]:
    """Convert any combination of Polars Series / NumPy arrays to NumPy float64 arrays."""
    return tuple(
        to_numpy_series(a) if isinstance(a, pl.Series) else np.asarray(a, dtype=np.float64)
        for a in arrays
    )
