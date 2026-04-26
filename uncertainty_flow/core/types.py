"""Core type aliases and constants for uncertainty_flow."""

from collections.abc import Iterator, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    pass

from .config import get_config


class _ConfigQuantiles(Sequence[float]):
    """Dynamic proxy that always reflects the active config quantiles."""

    def _values(self) -> tuple[float, ...]:
        return tuple(get_config().default_quantiles)

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return self._values()[index]

    def __len__(self) -> int:
        return len(self._values())

    def __iter__(self) -> Iterator[float]:
        return iter(self._values())

    def __contains__(self, value: object) -> bool:
        return value in self._values()

    def index(self, value: float, start: int = 0, stop: int | None = None) -> int:
        values = self._values()
        stop_index = len(values) if stop is None else stop
        return values.index(value, start, stop_index)

    def __repr__(self) -> str:
        return repr(self._values())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return NotImplemented


# Backward compatibility alias - DEFAULT_QUANTILES comes from config dynamically.
# For new code, use get_config().default_quantiles instead.
DEFAULT_QUANTILES = _ConfigQuantiles()


class CalibrationMethod(str, Enum):
    """Calibration method options."""

    HOLDOUT = "holdout"
    CROSS = "cross"


class CorrelationMode(str, Enum):
    """Correlation mode options."""

    AUTO = "auto"
    INDEPENDENT = "independent"


# Backward compatibility: keep Literal types
CalibrationMethodLiteral = Literal["holdout", "cross"]
CorrelationModeLiteral = Literal["auto", "independent"]

# Supported input types
PolarsInput = pl.DataFrame | pl.LazyFrame

# Target specification
TargetSpec = str | list[str]
