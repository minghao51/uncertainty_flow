"""Core type aliases and constants for uncertainty_flow."""

from collections.abc import Sequence
from enum import Enum

import polars as pl

from .config import get_config


class CalibrationMethod(str, Enum):
    HOLDOUT = "holdout"
    CROSS = "cross"


class CorrelationMode(str, Enum):
    AUTO = "auto"
    INDEPENDENT = "independent"


PolarsInput = pl.DataFrame | pl.LazyFrame
TargetSpec = str | list[str]


class _ConfigQuantiles(Sequence[float]):
    """Dynamic proxy reflecting the active config quantiles."""

    def _values(self) -> tuple[float, ...]:
        return tuple(get_config().default_quantiles)

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return self._values()[index]

    def __len__(self) -> int:
        return len(self._values())

    def __contains__(self, value: object) -> bool:
        return value in self._values()

    def __repr__(self) -> str:
        return repr(self._values())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return NotImplemented


DEFAULT_QUANTILES = _ConfigQuantiles()
