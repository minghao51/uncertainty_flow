"""DistributionPrediction - core output object for all models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ..utils.exceptions import InvalidDataError, QuantileError
from ..utils.polars_bridge import to_numpy_series
from .distribution_bayesian import BayesianMixin
from .distribution_causal import CausalMixin
from .distribution_groups import GroupMixin

if TYPE_CHECKING:
    from ..multivariate.copula import BaseCopula
    from .parametric import ParametricDistribution

MAX_SAMPLE_CHUNK_SIZE = 100_000
MAX_TOTAL_SAMPLES = 10_000_000
_QUANTILE_DISTANCE_THRESHOLD = 0.05


@dataclass
class _ScoringData:
    quantile_values: np.ndarray
    quantile_levels: list[float]
    targets: list[str] | None
    n_samples: int


def _warn_if_far(levels: np.ndarray, requested: dict[float, int]) -> None:
    import warnings

    far = []
    for target_level, idx in requested.items():
        actual = levels[idx]
        if abs(actual - target_level) > _QUANTILE_DISTANCE_THRESHOLD:
            far.append(f"{target_level:.2f}\u2192{actual:.2f}")
    if far:
        warnings.warn(
            f"Nearest quantile levels differ significantly from requested: "
            f"{', '.join(far)}. Consider using more quantile levels for "
            f"accurate summary statistics.",
            UserWarning,
            stacklevel=4,
        )


class DistributionPrediction(BayesianMixin, CausalMixin, GroupMixin):
    """
    Holds predicted distributions for N samples.

    Internal storage: NumPy arrays for efficiency.
    External interface: Polars DataFrames/Series.
    """

    def __init__(
        self,
        quantile_matrix: np.ndarray,
        quantile_levels: list[float],
        target_names: list[str],
        posterior: np.ndarray | dict[str, np.ndarray] | None = None,
        posterior_chains: dict[str, np.ndarray] | None = None,
        posterior_predictive: np.ndarray | None = None,
        group_predictions: dict[str, "DistributionPrediction"] | None = None,
        treatment_info: dict | None = None,
        copula: "BaseCopula | None" = None,
    ):
        if not np.all(np.isfinite(quantile_matrix)):
            raise InvalidDataError("quantile_matrix contains NaN or Inf values")

        if quantile_matrix.ndim != 2:
            raise InvalidDataError(f"quantile_matrix must be 2D, got shape {quantile_matrix.shape}")

        if quantile_matrix.shape[0] == 0:
            raise InvalidDataError("quantile_matrix must have at least one row")

        if len(target_names) == 0:
            raise InvalidDataError("target_names cannot be empty")

        n_targets = len(target_names)
        expected_cols = n_targets * len(quantile_levels)

        if quantile_matrix.shape[1] != expected_cols:
            raise InvalidDataError(
                f"quantile_matrix has {quantile_matrix.shape[1]} columns "
                f"but expected {expected_cols} columns for {n_targets} target(s) "
                f"with {len(quantile_levels)} quantile levels each"
            )

        self._quantiles = quantile_matrix
        self._levels = np.array(quantile_levels)
        self._targets = target_names
        self._n_samples = quantile_matrix.shape[0]
        self._n_quantiles = len(quantile_levels)

        self._posterior = posterior
        self._posterior_chains = posterior_chains
        self._posterior_predictive = posterior_predictive
        self._group_predictions = group_predictions
        self._treatment_info = treatment_info
        self._copula = copula

    def quantile(self, q: float | list[float]) -> pl.DataFrame:
        """Compute quantiles of the predicted distribution.

        Args:
            q: Quantile level(s) in [0, 1].

        Returns:
            DataFrame with one column per requested quantile.
        """
        if isinstance(q, (int, float, np.integer, np.floating)):
            q = [q]

        indices = [self._find_nearest_quantile_index(level) for level in q]

        if len(self._targets) == 1:
            columns = [f"q_{level:.3f}" for level in q]
            data = self._quantiles[:, indices]
        else:
            columns = [f"{target}_q_{level:.3f}" for target in self._targets for level in q]
            data = np.column_stack(
                [
                    self._quantile_slice(t_idx)[:, q_idx]
                    for t_idx in range(len(self._targets))
                    for q_idx in indices
                ]
            )

        return pl.DataFrame(data, schema=columns, orient="row")

    def interval(self, confidence: float = 0.9) -> pl.DataFrame:
        """Compute prediction intervals at the given confidence level.

        Args:
            confidence: Coverage probability in (0, 1).

        Returns:
            DataFrame with ``lower`` and ``upper`` columns.
        """
        if not (0 < confidence < 1):
            raise QuantileError(f"confidence must be in (0, 1), got {confidence}")

        alpha = (1 - confidence) / 2
        lower_idx = self._find_nearest_quantile_index(alpha)
        upper_idx = self._find_nearest_quantile_index(1 - alpha)

        if len(self._targets) == 1:
            columns = ["lower", "upper"]
            data = np.column_stack(
                [
                    self._quantiles[:, lower_idx],
                    self._quantiles[:, upper_idx],
                ]
            )
        else:
            columns = []
            data_parts = []
            for t_idx, target in enumerate(self._targets):
                columns.append(f"{target}_lower")
                columns.append(f"{target}_upper")
                q_slice = self._quantile_slice(t_idx)
                data_parts.append(q_slice[:, lower_idx])
                data_parts.append(q_slice[:, upper_idx])
            data = np.column_stack(data_parts)

        return pl.DataFrame(data, schema=columns, orient="row")

    def interval_bounds(
        self,
        confidence: float,
        target: str | None = None,
    ) -> tuple[pl.Series, pl.Series]:
        interval_df = self.interval(confidence)
        if "lower" in interval_df.columns:
            return interval_df["lower"], interval_df["upper"]
        t = target or self._targets[0]
        return interval_df[f"{t}_lower"], interval_df[f"{t}_upper"]

    def median(self) -> pl.Series | pl.DataFrame:
        """Compute the median of the predicted distribution.

        Returns:
            Series (single target) or DataFrame (multi-target) of median values.
        """
        median_idx = self._find_nearest_quantile_index(0.5)

        if len(self._targets) == 1:
            return pl.Series("median", self._quantiles[:, median_idx])
        else:
            data = np.column_stack(
                [self._quantile_slice(t_idx)[:, median_idx] for t_idx in range(len(self._targets))]
            )
            return pl.DataFrame(data, schema=self._targets, orient="row")

    def mean(self) -> pl.Series | pl.DataFrame:
        """Compute the mean of the predicted distribution via numerical integration.

        Returns:
            Series (single target) or DataFrame (multi-target) of mean values.
        """
        weights = np.diff(self._levels, prepend=0.0)
        if len(self._targets) == 1:
            values = self._quantiles[:, : self._n_quantiles] @ weights
            return pl.Series("mean", values)
        data = np.column_stack(
            [self._quantile_slice(t_idx) @ weights for t_idx in range(len(self._targets))]
        )
        return pl.DataFrame(data, schema=self._targets, orient="row")

    def std(self) -> pl.Series | pl.DataFrame:
        """Compute the standard deviation of the predicted distribution.

        Returns:
            Series (single target) or DataFrame (multi-target) of std values.
        """
        mean_vals = self.mean()
        if isinstance(mean_vals, pl.Series):
            mean_arr = mean_vals.to_numpy()
        else:
            mean_arr = mean_vals.to_numpy()

        if len(self._targets) == 1:
            q = self._quantiles[:, : self._n_quantiles]
            variance = (q**2) @ np.diff(self._levels, prepend=0.0) - mean_arr**2
            return pl.Series("std", np.sqrt(np.maximum(variance, 0.0)))

        var_cols = []
        for t_idx in range(len(self._targets)):
            q = self._quantile_slice(t_idx)
            m = mean_arr[:, t_idx]
            var = (q**2) @ np.diff(self._levels, prepend=0.0) - m**2
            var_cols.append(np.sqrt(np.maximum(var, 0.0)))
        data = np.column_stack(var_cols)
        return pl.DataFrame(data, schema=self._targets, orient="row")

    def crps(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
    ) -> float | dict[str, float]:
        """Compute the Continuous Ranked Probability Score.

        Args:
            y_true: Observed values.

        Returns:
            CRPS value (single target) or dict keyed by target name.
        """
        from .distribution_scoring import crps_score

        return crps_score(self, y_true)

    @staticmethod
    def _forward_cdf(
        quantile_values: np.ndarray,
        levels: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        n, k = quantile_values.shape
        if k == 1:
            return np.where(y <= quantile_values[:, 0], levels[0], 1.0)

        j = np.sum(quantile_values <= y[:, None], axis=1) - 1
        j = np.clip(j, 0, k - 2)

        row_idx = np.arange(n)
        q_j = quantile_values[row_idx, j]
        q_j1 = quantile_values[row_idx, j + 1]
        denom = q_j1 - q_j
        zero_denom = denom == 0
        denom[zero_denom] = 1.0
        frac = (y - q_j) / denom
        pit = levels[j] + frac * (levels[j + 1] - levels[j])
        pit[zero_denom] = (levels[j[zero_denom]] + levels[j[zero_denom] + 1]) / 2.0

        below = y <= quantile_values[:, 0]
        q0_eq_q1 = quantile_values[:, 0] == quantile_values[:, 1]
        below_denom = np.where(q0_eq_q1, 1.0, quantile_values[:, 1] - quantile_values[:, 0])
        below_frac = np.where(
            q0_eq_q1,
            levels[0] * 0.5,
            levels[0] + ((y - quantile_values[:, 0]) / below_denom) * levels[0],
        )
        pit[below] = below_frac[below]

        above = y >= quantile_values[:, -1]
        above = above & ~below
        qn_eq_qn1 = quantile_values[:, -1] == quantile_values[:, -2]
        above_denom = np.where(qn_eq_qn1, 1.0, quantile_values[:, -1] - quantile_values[:, -2])
        above_frac = np.where(
            qn_eq_qn1,
            1.0 - (1.0 - levels[-1]) * 0.5,
            levels[-1] + ((y - quantile_values[:, -1]) / above_denom) * (1.0 - levels[-1]),
        )
        pit[above] = above_frac[above]

        return np.clip(pit, 0.0, 1.0)

    def _pit_values(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if self._n_quantiles < 2:
            raise InvalidDataError(
                "PIT requires at least 2 quantile levels, "
                f"but DistributionPrediction has {self._n_quantiles}"
            )

        y_arr = self._coerce_y_true(y_true)

        if len(self._targets) == 1:
            return self._forward_cdf(self._quantiles, self._levels, y_arr.ravel())

        result = {}
        for t_idx, target in enumerate(self._targets):
            q_slice = self._quantile_slice(t_idx)
            result[target] = self._forward_cdf(
                q_slice, self._levels, self._target_truth(y_arr, t_idx)
            )
        return result

    def pit_histogram(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        n_bins: int = 10,
        chi2_test: bool = False,
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Probability Integral Transform histogram for calibration assessment.

        Args:
            y_true: Observed values.
            n_bins: Number of histogram bins.
            chi2_test: If True, add a chi-squared uniformity p-value column.

        Returns:
            DataFrame with bin counts and expected counts (single target),
            or dict keyed by target name (multi-target).
        """
        pit = self._pit_values(y_true)

        if isinstance(pit, dict):
            return {
                t: self._build_pit_hist(arr, n_bins, chi2_test=chi2_test) for t, arr in pit.items()
            }

        return self._build_pit_hist(pit, n_bins, chi2_test=chi2_test)

    @staticmethod
    def _build_pit_hist(
        pit_values: np.ndarray,
        n_bins: int,
        chi2_test: bool = False,
    ) -> pl.DataFrame:
        n = len(pit_values)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        counts, _ = np.histogram(pit_values, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        expected = np.full(n_bins, n / n_bins)

        data: dict[str, Any] = {
            "bin_center": centers,
            "count": counts.astype(float),
            "expected": expected,
        }

        if chi2_test:
            if n > 0 and np.all(expected > 0):
                from scipy.stats import chisquare

                _, pvalue = chisquare(counts, expected)
                data["chi2_pvalue"] = np.full(n_bins, float(pvalue))
            else:
                data["chi2_pvalue"] = np.full(n_bins, np.nan)

        return pl.DataFrame(data)

    def calibration_curve(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        n_bins: int = 20,
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Compute calibration curve (observed vs expected coverage).

        Args:
            y_true: Observed values.
            n_bins: Number of cumulative bins.

        Returns:
            DataFrame with ``expected_coverage``, ``observed_coverage``, and
            ``bin_center`` columns.
        """
        pit = self._pit_values(y_true)

        if isinstance(pit, dict):
            return {t: self._build_cal_curve(arr, n_bins) for t, arr in pit.items()}

        return self._build_cal_curve(pit, n_bins)

    @staticmethod
    def _build_cal_curve(pit_values: np.ndarray, n_bins: int) -> pl.DataFrame:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        counts, _ = np.histogram(pit_values, bins=edges)
        observed = np.cumsum(counts).astype(float) / len(pit_values)
        expected = np.cumsum(np.diff(edges))
        return pl.DataFrame(
            {
                "expected_coverage": expected,
                "observed_coverage": observed,
                "bin_center": centers,
            }
        )

    def plot_pit(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        n_bins: int = 10,
    ) -> None:
        from ..viz._plotting import plot_pit as _plot_pit

        _plot_pit(self, y_true, n_bins=n_bins)

    def _coerce_y_true(self, y_true: pl.Series | pl.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(y_true, pl.DataFrame):
            cols = [y_true[t].to_numpy() for t in self._targets]
            return np.column_stack(cols)
        if isinstance(y_true, pl.Series):
            return to_numpy_series(y_true)
        return np.asarray(y_true, dtype=np.float64)

    def sample(self, n: int, random_state: int | None = None) -> pl.DataFrame:
        """Draw random samples from the predicted distribution via inverse CDF.

        Args:
            n: Number of samples per input row.
            random_state: Optional RNG seed for reproducibility.

        Returns:
            DataFrame with a ``sample_id`` column plus one column per target.
        """
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise InvalidDataError(f"n must be a positive integer, got {n}")

        total_samples = self._n_samples * n
        if total_samples > MAX_TOTAL_SAMPLES:
            raise InvalidDataError(
                f"Total samples ({total_samples:,}) exceeds maximum ({MAX_TOTAL_SAMPLES:,}). "
                f"Reduce n (currently {n}) or number of input rows ({self._n_samples})."
            )

        rng = np.random.default_rng(random_state)

        if self._copula is not None and len(self._targets) > 1:
            return self._sample_in_chunks(n, rng, self._sample_joint_chunk)

        return self._sample_in_chunks(n, rng, self._sample_chunk)

    def _marginal_quantiles(self) -> np.ndarray:
        return self._quantiles.reshape(self._n_samples, len(self._targets), self._n_quantiles)

    def _sample_joint_chunk(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> pl.DataFrame:
        joint_samples = self._copula.sample(
            self._marginal_quantiles(),
            n_samples=n,
            quantile_levels=self._levels,
            random_state=rng,
        )

        if joint_samples.ndim == 2:
            sample_matrix = joint_samples
        else:
            sample_matrix = joint_samples.reshape(self._n_samples * n, len(self._targets))

        sample_ids = np.repeat(np.arange(self._n_samples), n)
        result = pl.DataFrame(sample_matrix, schema=self._targets, orient="row")
        result.insert_column(0, pl.Series("sample_id", sample_ids))
        return result

    def _sample_in_chunks(
        self,
        n: int,
        rng: np.random.Generator,
        chunk_fn,
    ) -> pl.DataFrame:
        if n <= MAX_SAMPLE_CHUNK_SIZE:
            return chunk_fn(n, rng)
        chunks = []
        remaining = n
        while remaining > 0:
            chunk_size = min(remaining, MAX_SAMPLE_CHUNK_SIZE)
            chunks.append(chunk_fn(chunk_size, rng))
            remaining -= chunk_size
        return pl.concat(chunks)

    def _sample_chunk(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> pl.DataFrame:
        uniform_samples = rng.uniform(0, 1, size=(self._n_samples, n))
        uniform_clipped = np.clip(uniform_samples, self._levels[0], self._levels[-1])

        target_samples_list = [
            self._vectorized_inverse_cdf(
                self._quantile_slice(t_idx), uniform_clipped, self._levels
            ).flatten()
            for t_idx in range(len(self._targets))
        ]

        sample_matrix = np.column_stack(target_samples_list)
        sample_ids = np.repeat(np.arange(self._n_samples), n)

        result = pl.DataFrame(sample_matrix, schema=self._targets, orient="row")
        result.insert_column(0, pl.Series("sample_id", sample_ids))
        return result

    @staticmethod
    def _vectorized_inverse_cdf(
        quantile_values: np.ndarray,
        uniform_clipped: np.ndarray,
        levels: np.ndarray,
    ) -> np.ndarray:
        lower_idx = np.searchsorted(levels, uniform_clipped, side="right") - 1
        lower_idx = np.clip(lower_idx, 0, len(levels) - 2)
        upper_idx = lower_idx + 1

        row_idx = np.arange(quantile_values.shape[0])[:, None]
        lower_x = levels[lower_idx]
        upper_x = levels[upper_idx]
        lower_y = quantile_values[row_idx, lower_idx]
        upper_y = quantile_values[row_idx, upper_idx]

        denom = upper_x - lower_x
        denom[denom == 0] = 1.0
        weight = (uniform_clipped - lower_x) / denom
        return lower_y + weight * (upper_y - lower_y)

    def plot(
        self,
        actuals: pl.Series | pl.DataFrame | None = None,
        confidence_bands: list[float] | None = None,
        title: str | None = None,
        targets: str | list[str] = "all",
        max_targets: int = 6,
    ) -> None:
        from ..viz._plotting import plot as _plot

        _plot(
            self,
            actuals=actuals,
            confidence_bands=confidence_bands,
            title=title,
            targets=targets,
            max_targets=max_targets,
        )

    @lru_cache(maxsize=128)
    def _find_nearest_quantile_index(self, q: float) -> int:
        distances = np.abs(self._levels - q)
        return int(np.argmin(distances))

    def _quantile_slice(self, target_idx: int) -> np.ndarray:
        q_start = target_idx * self._n_quantiles
        return self._quantiles[:, q_start : q_start + self._n_quantiles]

    def _interval_columns(self, target: str) -> tuple[str, str]:
        if len(self._targets) == 1:
            return "lower", "upper"
        return f"{target}_lower", f"{target}_upper"

    def _target_truth(self, y_arr: np.ndarray, t_idx: int, i: int | None = None) -> np.ndarray:
        if y_arr.ndim == 1:
            return y_arr
        if i is not None:
            return np.array([y_arr[i, t_idx]])
        return y_arr[:, t_idx]

    def _scoring_data(self) -> _ScoringData:
        return _ScoringData(
            quantile_values=self._quantiles,
            quantile_levels=self._levels.tolist(),
            targets=self._targets,
            n_samples=self._n_samples,
        )

    def __repr__(self) -> str:
        parts = [
            f"DistributionPrediction(n={self._n_samples}, "
            f"targets={self._targets}, quantiles={len(self._levels)}"
        ]
        if self._posterior is not None:
            parts.append(", posterior=True")
        if self._posterior_predictive is not None:
            parts.append(", posterior_predictive=True")
        parts.append(")")
        return "".join(parts)

    def uncertainty_decomposition(
        self,
        confidence: float = 0.9,
    ) -> dict[str, float]:
        return self._decomposition_for_target(0, confidence)

    def _decomposition_for_target(
        self, target_idx: int, confidence: float, interval: pl.DataFrame | None = None
    ) -> dict[str, float]:
        if interval is None:
            interval = self.interval(confidence)
        target = self._targets[target_idx]
        lower_col, upper_col = self._interval_columns(target)

        lower = to_numpy_series(interval[lower_col])
        upper = to_numpy_series(interval[upper_col])
        widths = upper - lower
        aleatoric = float(np.mean(widths))
        epistemic = float(np.var(widths))

        return {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": aleatoric + epistemic,
        }

    def summary(self, confidence: float = 0.9) -> pl.DataFrame:
        """Generate a summary of the prediction including median, interval widths, and
        uncertainty decomposition.

        Args:
            confidence: Confidence level for interval widths.

        Returns:
            DataFrame with one row per target.
        """
        median_idx = self._find_nearest_quantile_index(0.5)
        alpha = (1 - confidence) / 2
        lower_idx = self._find_nearest_quantile_index(alpha)
        upper_idx = self._find_nearest_quantile_index(1 - alpha)
        lower_50_idx = self._find_nearest_quantile_index(0.25)
        upper_50_idx = self._find_nearest_quantile_index(0.75)

        _warn_if_far(
            self._levels,
            {
                0.5: median_idx,
                alpha: lower_idx,
                1 - alpha: upper_idx,
                0.25: lower_50_idx,
                0.75: upper_50_idx,
            },
        )

        rows = []
        interval = self.interval(confidence)
        for t_idx, target in enumerate(self._targets):
            q_slice = self._quantile_slice(t_idx)

            median_val = float(np.mean(q_slice[:, median_idx]))
            width = float(np.mean(q_slice[:, upper_idx] - q_slice[:, lower_idx]))
            narrow = float(np.mean(q_slice[:, upper_50_idx] - q_slice[:, lower_50_idx]))

            decomp = self._decomposition_for_target(t_idx, confidence, interval)

            rows.append(
                {
                    "target": target,
                    "median": median_val,
                    "mean_width_90": width,
                    "mean_width_50": narrow,
                    "aleatoric": decomp["aleatoric"],
                    "epistemic": decomp["epistemic"],
                    "total_uncertainty": decomp["total"],
                }
            )

        return pl.DataFrame(rows)

    def fit_distribution(
        self,
        family: str = "auto",
        row_index: int | None = None,
    ) -> ParametricDistribution | list[ParametricDistribution]:
        """Fit a parametric distribution to the quantile predictions.

        Args:
            family: Distribution family (``"auto"``, ``"normal"``, ``"t"``,
                ``"laplace"``, ``"skew_normal"``).
            row_index: Specific row to fit.  If *None*, fits to the
                average quantile curve across all rows.

        Returns:
            A single ``ParametricDistribution`` (single target) or a list
            of ``ParametricDistribution`` (multi-target).
        """
        from .parametric import fit_parametric

        if len(self._targets) == 1:
            if row_index is not None:
                qv = self._quantiles[row_index, : self._n_quantiles]
            else:
                qv = np.mean(self._quantiles[:, : self._n_quantiles], axis=0)
            return fit_parametric(qv, self._levels, family=family)

        results = []
        for t_idx in range(len(self._targets)):
            q_slice = self._quantile_slice(t_idx)
            if row_index is not None:
                qv = q_slice[row_index]
            else:
                qv = np.mean(q_slice, axis=0)
            results.append(fit_parametric(qv, self._levels, family=family))
        return results

    def log_score(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        family: str = "auto",
    ) -> float | dict[str, float]:
        """Compute the logarithmic (log) score.

        Args:
            y_true: Observed values.
            family: Parametric family for density estimation.

        Returns:
            Log score value (single target) or dict keyed by target name.
        """
        from .distribution_scoring import log_score as _log_score

        return _log_score(self, y_true, family=family)

    def energy_score(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        n_samples: int = 1000,
        random_state: int | None = None,
    ) -> float:
        """Compute the energy score for multivariate predictions.

        Args:
            y_true: Observed values (2-D for multi-target).
            n_samples: Monte-Carlo samples per observation.
            random_state: Optional RNG seed.

        Returns:
            Energy score value.
        """
        from .distribution_scoring import energy_score as _energy_score

        return _energy_score(self, y_true, n_samples=n_samples, random_state=random_state)

    def variogram_score(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        n_samples: int = 1000,
        p: float = 0.5,
        random_state: int | None = None,
    ) -> float:
        """Compute the variogram score for multivariate predictions.

        Args:
            y_true: Observed values (2-D for multi-target).
            n_samples: Monte-Carlo samples per observation.
            p: Variogram order parameter.
            random_state: Optional RNG seed.

        Returns:
            Variogram score value.
        """
        from .distribution_scoring import variogram_score as _variogram_score

        return _variogram_score(self, y_true, n_samples=n_samples, p=p, random_state=random_state)
