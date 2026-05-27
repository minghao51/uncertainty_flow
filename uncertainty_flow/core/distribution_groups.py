"""Group (multi-modal) mixin for DistributionPrediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl

from ..utils.exceptions import InvalidDataError

if TYPE_CHECKING:
    from .distribution import DistributionPrediction


class _GroupHost(Protocol):
    _group_predictions: dict[str, DistributionPrediction] | None


class GroupMixin:
    def group_uncertainty(self: _GroupHost) -> dict[str, float]:
        if self._group_predictions is None:
            raise InvalidDataError(
                "group_uncertainty() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        result = {}
        for name, pred in self._group_predictions.items():
            interval = pred.interval(0.9)
            lower = interval["lower"].to_numpy()
            upper = interval["upper"].to_numpy()
            result[name] = float(np.mean(upper - lower))
        return result

    def group_intervals(self: _GroupHost, confidence: float = 0.9) -> dict[str, pl.DataFrame]:
        if self._group_predictions is None:
            raise InvalidDataError(
                "group_intervals() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        return {name: pred.interval(confidence) for name, pred in self._group_predictions.items()}

    def cross_group_correlation(self: _GroupHost) -> np.ndarray:
        if self._group_predictions is None:
            raise InvalidDataError(
                "cross_group_correlation() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        medians = np.column_stack(
            [pred.median().to_numpy() for pred in self._group_predictions.values()]
        )
        return np.corrcoef(medians.T)  # type: ignore
