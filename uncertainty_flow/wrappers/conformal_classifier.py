"""Conformal classifier with Adaptive Prediction Sets (APS).

Implements Romano et al. (2020): constructs prediction sets with marginal
coverage guarantees for multi-class classification.

The APS procedure:
1. For each calibration sample, compute softmax probabilities.
2. Sort classes by descending probability and compute cumulative sums.
3. The APS score is the cumulative probability at which the true class
   is included (the "necessary" set size).
4. The threshold is the (1 - alpha)-quantile of APS scores on the
   calibration set.
5. At test time, include classes until cumulative probability exceeds
   the threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
from sklearn.base import BaseEstimator, clone

from ..core.prediction_set import PredictionSet
from ..core.types import PolarsInput
from ..utils.exceptions import ConfigurationError, error_model_not_fitted
from ..utils.polars_bridge import materialize_lazyframe, to_numpy

if TYPE_CHECKING:
    pass


class ConformalClassifier:
    """Conformal classifier with Adaptive Prediction Sets.

    Wraps any sklearn classifier with a ``predict_proba`` method.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import polars as pl
        >>> from uncertainty_flow.wrappers import ConformalClassifier
        >>>
        >>> df = pl.DataFrame({
        ...     "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        ...     "label": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        ... })
        >>> model = ConformalClassifier(
        ...     base_model=RandomForestClassifier(random_state=42),
        ...     coverage_target=0.9,
        ...     random_state=42,
        ... )
        >>> model.fit(df, target="label")
        >>> pred = model.predict(df)
        >>> pred.set(0)
        >>> pred.size
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        coverage_target: float = 0.9,
        calibration_size: float = 0.2,
        random_state: int | None = None,
    ):
        if not (0 < coverage_target < 1):
            raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}")
        if not (0 < calibration_size < 1):
            raise ValueError(f"calibration_size must be in (0, 1), got {calibration_size}")

        self.base_model = base_model
        self.coverage_target = coverage_target
        self.calibration_size = calibration_size
        self.random_state = random_state

        self._fitted = False
        self._model: BaseEstimator | None = None
        self._feature_cols_: list[str] = []
        self._target_col_: str = ""
        self._class_names_: list[str] = []
        self._threshold: float | None = None

    def fit(
        self,
        data: PolarsInput,
        target: str | None = None,
        **kwargs,
    ) -> ConformalClassifier:
        data = materialize_lazyframe(data)

        if target is None:
            raise ConfigurationError("target is required for ConformalClassifier")
        target_str = target if isinstance(target, str) else target[0]
        self._target_col_ = target_str

        if target_str not in data.columns:
            raise ValueError(
                f"Target column '{target_str}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        self._feature_cols_ = [c for c in data.columns if c != target_str]
        if not self._feature_cols_:
            raise ValueError("No feature columns found.")

        n = len(data)
        n_calib = max(1, int(n * self.calibration_size))
        n_train = n - n_calib

        rng = np.random.default_rng(self.random_state)
        shuffled_idx = rng.permutation(n)
        train_idx = shuffled_idx[:n_train]
        calib_idx = shuffled_idx[n_train:]

        train_data = data[train_idx.tolist()]
        calib_data = data[calib_idx.tolist()]

        x_train = to_numpy(train_data, self._feature_cols_)
        y_train = train_data[target_str].to_numpy().flatten()
        x_calib = to_numpy(calib_data, self._feature_cols_)
        y_calib = calib_data[target_str].to_numpy().flatten()

        self._model = clone(self.base_model)
        if self.random_state is not None and "random_state" in self._model.get_params(deep=False):
            self._model.set_params(random_state=self.random_state)
        self._model.fit(x_train, y_train)

        self._class_names_ = list(self._model.classes_)

        calib_probs = self._model.predict_proba(x_calib)
        miscoverage_alpha = 1.0 - self.coverage_target
        self._threshold = self._compute_aps_threshold(calib_probs, y_calib, miscoverage_alpha)

        self._fitted = True
        return self

    def predict(self, data: PolarsInput) -> PredictionSet:
        if not self._fitted:
            error_model_not_fitted("ConformalClassifier")
        if self._model is None:
            raise RuntimeError("Internal error: model is None after fit")
        if self._threshold is None:
            raise RuntimeError("Internal error: threshold is None after fit")

        data = materialize_lazyframe(data)
        x = to_numpy(data, self._feature_cols_)
        probs = self._model.predict_proba(x)

        class_sets = self._build_prediction_sets(probs, self._threshold, self._class_names_)

        return PredictionSet(
            class_sets=class_sets,
            class_names=self._class_names_,
            probabilities=probs,
            coverage_target=self.coverage_target,
            threshold=self._threshold,
        )

    def predict_batch(
        self,
        data: PolarsInput,
        batch_size: int = 1000,
    ) -> Iterator[PredictionSet]:
        """
        Generate prediction sets in chunks.

        Args:
            data: Feature DataFrame.
            batch_size: Rows per batch.

        Yields:
            ``PredictionSet`` per chunk.
        """
        data = materialize_lazyframe(data)
        n = len(data)
        for start in range(0, n, batch_size):
            yield self.predict(data[start : start + batch_size])

    def save(self, path: str | Path, **kwargs) -> None:
        """Persist the conformal classifier via the standard .uf archive."""
        from ..core._persistence import save_model_archive

        save_model_archive(self, path, **kwargs)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> ConformalClassifier:
        """Load a conformal classifier from a .uf archive."""
        from ..core._persistence import load_model_archive

        model, _ = load_model_archive(path, **kwargs)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model

    def _compute_aps_threshold(
        self,
        calib_probs: np.ndarray,
        y_calib: np.ndarray,
        miscoverage_alpha: float,
    ) -> float:
        """Compute APS threshold at the conformal level 1 - alpha."""
        classes = self._model.classes_
        class_to_idx = {c: i for i, c in enumerate(classes)}

        aps_scores = np.empty(len(y_calib))
        for i in range(len(y_calib)):
            probs = calib_probs[i]
            true_idx = class_to_idx.get(y_calib[i], -1)
            if true_idx < 0:
                raise ValueError(f"Calibration label '{y_calib[i]}' not in model classes")

            order = np.argsort(probs)[::-1]
            cumsum = 0.0
            for cls_idx in order:
                cumsum += probs[cls_idx]
                if cls_idx == true_idx:
                    aps_scores[i] = cumsum
                    break

        n = len(aps_scores)
        quantile_idx = int(np.ceil((n + 1) * (1.0 - miscoverage_alpha)))
        quantile_idx = min(quantile_idx, n) - 1
        quantile_idx = max(quantile_idx, 0)

        sorted_scores = np.sort(aps_scores)
        return float(sorted_scores[quantile_idx])

    @staticmethod
    def _build_prediction_sets(
        probs: np.ndarray,
        threshold: float,
        class_names: list[str],
    ) -> list[list[str]]:
        sets: list[list[str]] = []

        for i in range(probs.shape[0]):
            row = probs[i]
            order = np.argsort(row)[::-1]

            included: list[str] = []
            cumsum = 0.0
            for cls_idx in order:
                included.append(class_names[cls_idx])
                cumsum += row[cls_idx]
                if cumsum >= threshold:
                    break

            sets.append(included)

        return sets
