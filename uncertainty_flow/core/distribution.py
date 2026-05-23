"""DistributionPrediction - core output object for all models."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ..utils.exceptions import InvalidDataError, QuantileError
from ..utils.polars_bridge import to_numpy_series

if TYPE_CHECKING:
    from ..multivariate.copula import BaseCopula
    from .parametric import ParametricDistribution

# Constants
MAX_SAMPLE_CHUNK_SIZE = 100_000
MAX_TOTAL_SAMPLES = 10_000_000
_QUANTILE_DISTANCE_THRESHOLD = 0.05


def _warn_if_far(levels: np.ndarray, requested: dict[float, int]) -> None:
    import warnings

    far = []
    for target_level, idx in requested.items():
        actual = levels[idx]
        if abs(actual - target_level) > _QUANTILE_DISTANCE_THRESHOLD:
            far.append(f"{target_level:.2f}→{actual:.2f}")
    if far:
        warnings.warn(
            f"Nearest quantile levels differ significantly from requested: "
            f"{', '.join(far)}. Consider using more quantile levels for "
            f"accurate summary statistics.",
            UserWarning,
            stacklevel=4,
        )


class DistributionPrediction:
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

        # Optional extensions for Bayesian, Multi-Modal, Causal modules
        self._posterior = posterior
        self._posterior_chains = posterior_chains
        self._posterior_predictive = posterior_predictive
        self._group_predictions = group_predictions
        self._treatment_info = treatment_info
        self._copula = copula

    def quantile(self, q: float | list[float]) -> pl.DataFrame:
        """
        Extract specific quantile levels.

        Args:
            q: Single quantile level or list of levels

        Returns:
            Polars DataFrame with columns like "q_0.05" or "price_q_0.05" for multivariate
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
        """
        Return prediction interval.

        For 0.9 confidence: uses 0.05 and 0.95 quantiles.
        Returns columns: lower, upper (or price_lower, price_upper for multivariate)

        Args:
            confidence: Confidence level (e.g., 0.9 for 90% interval)

        Returns:
            Polars DataFrame with lower/upper bounds
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
        """Return (lower, upper) series for a prediction interval.

        Args:
            confidence: Confidence level (e.g., 0.9 for 90% interval).
            target: Target column name (required for multivariate; defaults
                to first target if *None*).

        Returns:
            Tuple of (lower, upper) Polars Series.
        """
        interval_df = self.interval(confidence)
        if "lower" in interval_df.columns:
            return interval_df["lower"], interval_df["upper"]
        t = target or self._targets[0]
        return interval_df[f"{t}_lower"], interval_df[f"{t}_upper"]

    def median(self) -> pl.Series | pl.DataFrame:
        """Return the 0.5 quantile as a point estimate."""
        median_idx = self._find_nearest_quantile_index(0.5)

        if len(self._targets) == 1:
            return pl.Series("median", self._quantiles[:, median_idx])
        else:
            data = np.column_stack(
                [self._quantile_slice(t_idx)[:, median_idx] for t_idx in range(len(self._targets))]
            )
            return pl.DataFrame(data, schema=self._targets, orient="row")

    def crps(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
    ) -> float | dict[str, float]:
        """
        Compute the exact CRPS from quantile predictions.

        Uses the quantile-score decomposition (Laio & Tamea 2007) — no
        Gaussian approximation.  Requires at least 2 quantile levels.

        Args:
            y_true: True values.  Polars Series (univariate), DataFrame
                (multivariate — one column per target), or numpy array.

        Returns:
            Float CRPS for univariate predictions, or ``{target: crps}`` dict
            for multivariate.
        """
        from ..metrics.crps import crps_quantile

        if self._n_quantiles < 2:
            raise InvalidDataError(
                "CRPS requires at least 2 quantile levels, "
                f"but DistributionPrediction has {self._n_quantiles}"
            )

        y_arr = self._coerce_y_true(y_true)

        if len(self._targets) == 1:
            return crps_quantile(y_arr, self._quantiles, self._levels)

        result = {}
        for t_idx, target in enumerate(self._targets):
            q_slice = self._quantile_slice(t_idx)
            result[target] = crps_quantile(
                self._target_truth(y_arr, t_idx),
                q_slice,
                self._levels,
            )
        return result

    @staticmethod
    def _forward_cdf(
        quantile_values: np.ndarray,
        levels: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the piecewise-linear CDF at observed values (vectorized).

        Args:
            quantile_values: (n_samples, n_quantiles) predicted quantile values.
            levels: (n_quantiles,) quantile levels.
            y: (n_samples,) true values.

        Returns:
            (n_samples,) PIT values in [0, 1].
        """
        n, k = quantile_values.shape
        if k == 1:
            return np.where(y <= quantile_values[:, 0], levels[0], 1.0)

        j = np.array(
            [np.searchsorted(quantile_values[i], y[i], side="right") - 1 for i in range(n)]
        )
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
        """
        Compute PIT values F_i(y_i) for each observation.

        Args:
            y_true: True values.

        Returns:
            (n,) array for univariate, or {target: array} for multivariate.
        """
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
        """
        Compute PIT histogram for calibration assessment.

        If forecasts are perfectly calibrated, PIT values ~ Uniform(0, 1),
        so each bin should contain roughly n / n_bins observations.

        Args:
            y_true: True values.
            n_bins: Number of histogram bins (default 10).
            chi2_test: If True, include a chi-squared uniformity test p-value
                as a ``chi2_pvalue`` column (default False).

        Returns:
            DataFrame with columns: bin_center, count, expected.
            If ``chi2_test=True``, also includes ``chi2_pvalue``.
            For multivariate, returns {target: DataFrame}.
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
            # Chi-squared test for uniformity
            # Avoid division by zero — expected is constant and > 0 when n > 0
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
        """
        Compute reliability diagram data (calibration curve).

        Bins PIT values and compares expected (nominal) coverage to observed
        (empirical) coverage at increasing probability thresholds.

        Args:
            y_true: True values.
            n_bins: Number of bins (default 20).

        Returns:
            DataFrame with columns: expected_coverage, observed_coverage.
            For multivariate, returns {target: DataFrame}.
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
        """Plot PIT histogram with uniform reference line. Requires matplotlib.

        Args:
            y_true: True values.
            n_bins: Number of histogram bins (default 10).
        """
        from ..viz._plotting import plot_pit as _plot_pit

        _plot_pit(self, y_true, n_bins=n_bins)

    def _coerce_y_true(self, y_true: pl.Series | pl.DataFrame | np.ndarray) -> np.ndarray:
        """Convert y_true input to a numpy array shaped for this prediction."""
        if isinstance(y_true, pl.DataFrame):
            cols = [y_true[t].to_numpy() for t in self._targets]
            return np.column_stack(cols)
        if isinstance(y_true, pl.Series):
            return to_numpy_series(y_true)
        return np.asarray(y_true, dtype=np.float64)

    def sample(self, n: int, random_state: int | None = None) -> pl.DataFrame:
        """
        Draw n samples per input row via piecewise-linear inverse CDF.

        For each row and each target, builds a CDF from the predicted quantile
        matrix (quantile values -> cumulative probability) and draws samples
        by inverting the CDF.

        Args:
            n: Number of samples to draw per input row.
            random_state: Optional random seed for reproducibility.

        Returns:
            Polars DataFrame with (n * n_samples) rows and columns:
            - sample_id: index of the original input row (0 to n_samples-1, repeated n times)
            - One column per target with sampled values

        Raises:
            InvalidDataError: If n is invalid or would exceed memory limits.
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
        """Reshape the flat quantile matrix into [row, target, quantile]."""
        return self._quantiles.reshape(self._n_samples, len(self._targets), self._n_quantiles)

    def _sample_joint_chunk(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> pl.DataFrame:
        """Sample jointly using a fitted copula."""
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
        """Generic chunked sampling — delegates to *chunk_fn* per chunk."""
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
        """Sample with a single chunk (n <= MAX_SAMPLE_CHUNK_SIZE)."""
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
        """
        Vectorized inverse CDF sampling via piecewise-linear interpolation.

        Args:
            quantile_values: (n_samples, n_quantiles) array of quantile values
            uniform_clipped: (n_samples, n) array of uniform samples
            levels: (n_quantiles,) array of quantile levels

        Returns:
            (n_samples, n) array of sampled values
        """
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
        """Fan chart of quantile bands. Requires matplotlib.

        Args:
            actuals: Optional actual values for comparison.
            confidence_bands: Confidence levels (default: [0.5, 0.8, 0.9, 0.95]).
            title: Optional plot title.
            targets: Target(s) to plot. ``"all"`` plots every target.
            max_targets: Maximum subplot panels (default 6).
        """
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
        """Find index of nearest quantile level. Cached for repeated lookups."""
        distances = np.abs(self._levels - q)
        return int(np.argmin(distances))

    def _quantile_slice(self, target_idx: int) -> np.ndarray:
        """Return (n_samples, n_quantiles) slice for a given target index."""
        q_start = target_idx * self._n_quantiles
        return self._quantiles[:, q_start : q_start + self._n_quantiles]

    def _interval_columns(self, target: str) -> tuple[str, str]:
        """Return (lower_col, upper_col) column name pair for a target."""
        if len(self._targets) == 1:
            return "lower", "upper"
        return f"{target}_lower", f"{target}_upper"

    def _target_truth(self, y_arr: np.ndarray, t_idx: int, i: int | None = None) -> np.ndarray:
        """Extract truth values for a single target from a full y array."""
        if y_arr.ndim == 1:
            return y_arr
        if i is not None:
            return np.array([y_arr[i, t_idx]])
        return y_arr[:, t_idx]

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

    # --- Bayesian posterior methods ---

    def posterior_samples(self) -> np.ndarray:
        """Return raw posterior parameter draws as a 2D matrix."""
        if self._posterior is None:
            raise InvalidDataError(
                "posterior_samples() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        if isinstance(self._posterior, np.ndarray):
            return self._posterior

        # Concatenate named posterior tensors into [draw, feature] for summaries.
        matrices = []
        for arr in self._posterior.values():
            arr_np = np.asarray(arr)
            if arr_np.ndim == 1:
                matrices.append(arr_np[:, np.newaxis])
            else:
                matrices.append(arr_np.reshape(arr_np.shape[0], -1))
        if not matrices:
            raise InvalidDataError("posterior data is empty")
        return np.column_stack(matrices)

    def posterior_parameter_interval(self, confidence: float = 0.9) -> pl.DataFrame:
        """Compute parameter credible intervals from posterior draws."""
        if not (0 < confidence < 1):
            raise QuantileError(f"confidence must be in (0, 1), got {confidence}")
        samples = self.posterior_samples()
        alpha = (1 - confidence) / 2
        lower = np.quantile(samples, alpha, axis=0)
        upper = np.quantile(samples, 1 - alpha, axis=0)
        return pl.DataFrame({"lower": lower, "upper": upper}, orient="row")

    def credible_interval(self, confidence: float = 0.9) -> pl.DataFrame:
        """Compute predictive credible intervals for each prediction row."""
        import warnings

        if not (0 < confidence < 1):
            raise QuantileError(f"confidence must be in (0, 1), got {confidence}")

        if self._posterior_predictive is None:
            warnings.warn(
                "credible_interval() currently falls back to parameter intervals when "
                "posterior predictive draws are unavailable. Use "
                "posterior_parameter_interval() for explicit parameter intervals.",
                FutureWarning,
                stacklevel=2,
            )
            return self.posterior_parameter_interval(confidence)

        alpha = (1 - confidence) / 2
        lower = np.quantile(self._posterior_predictive, alpha, axis=1)
        upper = np.quantile(self._posterior_predictive, 1 - alpha, axis=1)
        return pl.DataFrame({"lower": lower, "upper": upper}, orient="row")

    def rhat(self) -> np.ndarray:
        """Compute Gelman-Rubin R-hat convergence diagnostic from true chains."""
        if self._posterior_chains is None:
            raise InvalidDataError(
                "rhat() requires posterior chain data. "
                "Fit BayesianQuantileRegressor with num_chains > 1."
            )

        all_rhat = []
        for name, arr in self._posterior_chains.items():
            chains = np.asarray(arr)
            if chains.ndim < 2:
                raise InvalidDataError(f"posterior chain array for '{name}' must be at least 2D")
            n_chains = chains.shape[0]
            chain_len = chains.shape[1]
            if n_chains < 2:
                raise InvalidDataError(
                    f"rhat() requires at least 2 chains for '{name}', got {n_chains}."
                )
            if chain_len < 2:
                raise InvalidDataError(
                    f"rhat() requires at least 2 draws per chain for '{name}', got {chain_len}."
                )
            reshaped = chains.reshape(n_chains, chain_len, -1)
            chain_means = reshaped.mean(axis=1)
            b = chain_len * np.var(chain_means, axis=0, ddof=1)
            w = np.mean(np.var(reshaped, axis=1, ddof=1), axis=0)
            if np.any(w < 1e-10):
                raise InvalidDataError(
                    f"Within-chain variance too close to zero for R-hat calculation for '{name}'."
                )
            var_hat = (1 - 1 / chain_len) * w + (1 / chain_len) * b
            all_rhat.append(np.sqrt(var_hat / w))

        if not all_rhat:
            raise InvalidDataError(
                "rhat() requires at least one posterior chain parameter to evaluate."
            )
        return np.concatenate(all_rhat)

    def posterior_summary(self) -> pl.DataFrame:
        """Return summary statistics of posterior samples."""
        if self._posterior is None:
            raise InvalidDataError(
                "posterior_summary() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        samples = self.posterior_samples()
        return pl.DataFrame(
            {
                "mean": np.mean(samples, axis=0),
                "std": np.std(samples, axis=0),
                "q025": np.quantile(samples, 0.025, axis=0),
                "q50": np.quantile(samples, 0.5, axis=0),
                "q975": np.quantile(samples, 0.975, axis=0),
            }
        )

    # --- Multi-modal group methods ---

    def group_uncertainty(self) -> dict[str, float]:
        """Return per-group uncertainty contribution (interval width)."""
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

    def group_intervals(self, confidence: float = 0.9) -> dict[str, pl.DataFrame]:
        """Return per-group prediction intervals."""
        if self._group_predictions is None:
            raise InvalidDataError(
                "group_intervals() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        return {name: pred.interval(confidence) for name, pred in self._group_predictions.items()}

    def cross_group_correlation(self) -> np.ndarray:
        """Return cross-group correlation matrix based on group median predictions."""
        if self._group_predictions is None:
            raise InvalidDataError(
                "cross_group_correlation() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        medians = np.column_stack(
            [
                pred._quantiles[:, pred._find_nearest_quantile_index(0.5)]
                for pred in self._group_predictions.values()
            ]
        )
        return np.corrcoef(medians.T)  # type: ignore

    # --- Causal treatment methods ---

    def treatment_effect(self) -> np.ndarray:
        """Return CATE point estimates."""
        if self._treatment_info is None:
            raise InvalidDataError(
                "treatment_effect() requires treatment info. "
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        return self._treatment_info["cate"]  # type: ignore

    def average_treatment_effect(self) -> dict:
        """Return ATE with confidence interval."""
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

    def heterogeneity_score(self) -> float:
        """Return CATE variance as heterogeneity measure."""
        if self._treatment_info is None:
            raise InvalidDataError(
                "heterogeneity_score() requires treatment info. "
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        return float(np.var(self._treatment_info["cate"]))

    def uncertainty_decomposition(
        self,
        confidence: float = 0.9,
    ) -> dict[str, float]:
        """
        Return a lightweight heuristic uncertainty decomposition.
        Aleatoric uncertainty (data noise): Irreducible uncertainty inherent in the data.
        Epistemic uncertainty (model uncertainty): Reducible uncertainty due to limited
            data/knowledge.

        This method does not refit or evaluate an ensemble of models. It is a cheap
        summary derived from this single `DistributionPrediction` object:
        - Aleatoric: Average width of prediction intervals (data uncertainty)
        - Epistemic: Variance of interval widths across samples (model uncertainty)

        For model-based decomposition with bootstrap refits, use
        `uncertainty_flow.decomposition.EnsembleDecomposition`.

        Args:
            confidence: Confidence level for interval width calculation (default: 0.9)

        Returns:
            Dictionary with:
                - aleatoric: Irreducible uncertainty (average interval width)
                - epistemic: Heuristic uncertainty summary (variance of interval widths)
                - total: Combined uncertainty

        Examples
        --------
        >>> pred = model.predict(X_test)
        >>> decomposition = pred.uncertainty_decomposition()
        >>> print(f"Total: {decomposition['total']:.2f}")
        >>> print(f"  Aleatoric: {decomposition['aleatoric']:.2f}")
        >>> print(f"  Epistemic: {decomposition['epistemic']:.2f}")
        """
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
        """
        One-row-per-target overview of the prediction distribution.

        Columns: target, median, mean_width_90, mean_width_50,
        aleatoric, epistemic, total_uncertainty.

        ``mean_width_90`` is the mean width at the ``confidence`` level.
        ``mean_width_50`` is the mean inter-quartile range (25th–75th percentile).

        Args:
            confidence: Confidence level for the primary interval width (default 0.9).

        Returns:
            Polars DataFrame with one row per target.
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
        """
        Fit a parametric distribution to the quantile predictions.

        For univariate predictions, fits a single distribution. For
        multivariate, fits one distribution per target.

        Args:
            family: One of ``"normal"``, ``"student_t"``, ``"lognormal"``,
                ``"gamma"``, or ``"auto"`` (best fit by KS distance).
            row_index: If given, fit only for that row.  Otherwise fit for
                the *mean* quantile vector across all rows.

        Returns:
            A single ``ParametricDistribution`` for univariate, or a list
            of ``ParametricDistribution`` (one per target) for multivariate.
            When ``row_index`` is given, always returns a single distribution
            for univariate or a list for multivariate.
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
        """
        Compute the mean negative log-likelihood (log-score).

        Fits a parametric distribution from the predicted quantiles, then
        evaluates the log-density at the true values.  Higher is better
        (less negative).

        Args:
            y_true: True values.
            family: Distribution family for fitting, or ``"auto"``.

        Returns:
            Mean log-score (float) for univariate, or ``{target: score}`` dict
            for multivariate.
        """
        from ..metrics.log_score import log_score as _log_score

        y_arr = self._coerce_y_true(y_true)

        if len(self._targets) == 1:
            return _log_score(
                y_arr, self._quantiles[:, : self._n_quantiles], self._levels, family=family
            )

        result = {}
        for t_idx, target in enumerate(self._targets):
            result[target] = _log_score(
                self._target_truth(y_arr, t_idx),
                self._quantile_slice(t_idx),
                self._levels,
                family=family,
            )
        return result

    def energy_score(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        n_samples: int = 1000,
        random_state: int | None = None,
    ) -> float:
        """
        Compute the energy score for multivariate predictions.

        A proper scoring rule that generalises CRPS to the multivariate
        case.  Requires at least 2 targets.

        Args:
            y_true: True values (array with columns matching targets).
            n_samples: Monte Carlo samples per observation.
            random_state: Random seed.

        Returns:
            Mean energy score (float).
        """
        from ..metrics.multivariate import energy_score as _energy_score

        return _energy_score(self, y_true, n_samples=n_samples, random_state=random_state)

    def variogram_score(
        self,
        y_true: pl.Series | pl.DataFrame | np.ndarray,
        n_samples: int = 1000,
        p: float = 0.5,
        random_state: int | None = None,
    ) -> float:
        """
        Compute the variogram score for multivariate predictions.

        A proper scoring rule sensitive to the correlation structure.
        Requires at least 2 targets.

        Args:
            y_true: True values (array with columns matching targets).
            n_samples: Monte Carlo samples per observation.
            p: Power parameter (default 0.5).
            random_state: Random seed.

        Returns:
            Mean variogram score (float).
        """
        from ..metrics.multivariate import variogram_score as _variogram_score

        return _variogram_score(self, y_true, n_samples=n_samples, p=p, random_state=random_state)
