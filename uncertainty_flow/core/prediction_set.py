"""PredictionSet — result object for conformal classification.

Analogue of DistributionPrediction for classification tasks.
Stores the predicted class sets with calibrated coverage guarantees.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..utils.exceptions import InvalidDataError


class PredictionSet:
    """Prediction set for conformal classification.

    For each sample, stores the set of classes included at the calibrated
    coverage level. Analogous to ``DistributionPrediction`` for regression.

    Args:
        class_sets: List of lists — each inner list contains the class labels
            included in that sample's prediction set.
        class_names: Ordered list of all class names.
        probabilities: (n_samples, n_classes) matrix of softmax probabilities.
        coverage_target: The target marginal coverage level.
        threshold: The APS threshold used to construct the sets.
    """

    def __init__(
        self,
        class_sets: list[list[str]],
        class_names: list[str],
        probabilities: np.ndarray,
        coverage_target: float,
        threshold: float,
    ):
        if not class_sets or not class_names:
            raise InvalidDataError("class_sets and class_names must be non-empty")
        if len(class_sets) != probabilities.shape[0]:
            raise InvalidDataError(
                f"class_sets length ({len(class_sets)}) "
                f"!= probabilities rows ({probabilities.shape[0]})"
            )
        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
            raise InvalidDataError("probabilities must not contain NaN or Inf")
        if not (0 < coverage_target < 1):
            raise InvalidDataError("coverage_target must be in (0, 1)")
        if threshold < 0:
            raise InvalidDataError("threshold must be non-negative")

        self._class_sets = class_sets
        self._class_names = class_names
        self._probabilities = probabilities
        self._coverage_target = coverage_target
        self._threshold = threshold
        self._n_samples = len(class_sets)
        self._n_classes = len(class_names)

    def prediction_sets(self, sample_index: int | None = None) -> list[str] | list[list[str]]:
        """Return the prediction set for one or all samples.

        Args:
            sample_index: Index of a single sample, or ``None`` for all.

        Returns:
            Single list of class labels, or list of lists for all samples.
        """
        if sample_index is not None:
            return self._class_sets[sample_index]
        return self._class_sets

    def set(self, sample_index: int | None = None) -> list[str] | list[list[str]]:
        """Backward-compatible alias for :meth:`prediction_sets`."""
        return self.prediction_sets(sample_index)

    @property
    def coverage(self) -> float:
        """Return the theoretical target coverage level."""
        return self._coverage_target

    @property
    def size(self) -> float:
        """Return the average set size across all samples."""
        return float(np.mean([len(s) for s in self._class_sets]))

    def size_by_sample(self) -> list[int]:
        """Return the set size for each sample."""
        return [len(s) for s in self._class_sets]

    def probabilities(self) -> pl.DataFrame:
        """Return the softmax probability matrix as a Polars DataFrame.

        Returns:
            DataFrame with columns ``class_<name>``.
        """
        schema = [f"class_{c}" for c in self._class_names]
        return pl.DataFrame(self._probabilities, schema=schema, orient="row")

    def summary(self) -> pl.DataFrame:
        """Return a one-row summary of the prediction set.

        Columns: coverage_target, avg_set_size, n_samples, n_classes.
        """
        return pl.DataFrame(
            [
                {
                    "coverage_target": self._coverage_target,
                    "avg_set_size": self.size,
                    "n_samples": self._n_samples,
                    "n_classes": self._n_classes,
                }
            ]
        )

    def __repr__(self) -> str:
        return (
            f"PredictionSet(n_samples={self._n_samples}, "
            f"n_classes={self._n_classes}, "
            f"coverage={self._coverage_target:.2f}, "
            f"avg_size={self.size:.2f})"
        )
