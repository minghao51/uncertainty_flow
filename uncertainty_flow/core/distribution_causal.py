"""Causal treatment effect mixin for DistributionPrediction."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..utils.exceptions import InvalidDataError


class _CausalHost(Protocol):
    _treatment_info: dict | None


class CausalMixin:
    def treatment_effect(self: _CausalHost) -> np.ndarray:
        if self._treatment_info is None:
            raise InvalidDataError(
                "treatment_effect() requires treatment info. "
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        return self._treatment_info["cate"]  # type: ignore

    def average_treatment_effect(self: _CausalHost) -> dict:
        if self._treatment_info is None:
            raise InvalidDataError(
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        if "ate" not in self._treatment_info or "ate_ci" not in self._treatment_info:
            raise InvalidDataError(
                "average_treatment_effect() requires evaluated treatment metrics. "
                "Call CausalUncertaintyEstimator.evaluate(...) on labeled data first."
            )
        return {
            "ate": self._treatment_info["ate"],
            "ci": self._treatment_info["ate_ci"],
        }

    def heterogeneity_score(self: _CausalHost) -> float:
        if self._treatment_info is None:
            raise InvalidDataError(
                "heterogeneity_score() requires treatment info. "
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        return float(np.var(self._treatment_info["cate"]))
