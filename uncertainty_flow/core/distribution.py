"""DistributionPrediction - core output object for all models."""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..utils.exceptions import InvalidDataError, error_invalid_data, error_quantile_invalid

if TYPE_CHECKING:
    pass

# Constants
MAX_SAMPLE_CHUNK_SIZE = 100_000
MAX_TOTAL_SAMPLES = 10_000_000
PLOT_MAX_SAMPLES = 500


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
        posterior: np.ndarray | None = None,
        group_predictions: dict[str, "DistributionPrediction"] | None = None,
        treatment_info: dict | None = None,
    ):
        # Validate inputs
        if not np.all(np.isfinite(quantile_matrix)):
            error_invalid_data("quantile_matrix contains NaN or Inf values")

        if quantile_matrix.ndim != 2:
            error_invalid_data(f"quantile_matrix must be 2D, got shape {quantile_matrix.shape}")

        if quantile_matrix.shape[0] == 0:
            error_invalid_data("quantile_matrix must have at least one row")

        if len(target_names) == 0:
            error_invalid_data("target_names cannot be empty")

        # For multivariate: matrix has n_targets * n_quantiles columns
        # For univariate: matrix has n_quantiles columns
        n_targets = len(target_names)
        expected_cols = n_targets * len(quantile_levels)

        if quantile_matrix.shape[1] != expected_cols:
            error_invalid_data(
                f"quantile_matrix has {quantile_matrix.shape[1]} columns "
                f"but expected {expected_cols} columns for {n_targets} target(s) "
                f"with {len(quantile_levels)} quantile levels each"
            )

        # Store internally as NumPy
        self._quantiles = quantile_matrix
        self._levels = np.array(quantile_levels)
        self._targets = target_names
        self._n_samples = quantile_matrix.shape[0]
        self._n_quantiles = len(quantile_levels)

        # Optional extensions for Bayesian, Multi-Modal, Causal modules
        self._posterior = posterior
        self._group_predictions = group_predictions
        self._treatment_info = treatment_info

    def quantile(self, q: float | list[float]) -> pl.DataFrame:
        """
        Extract specific quantile levels.

        Args:
            q: Single quantile level or list of levels

        Returns:
            Polars DataFrame with columns like "q_0.05" or "price_q_0.05" for multivariate
        """
        if isinstance(q, (int, float)):
            q = [q]

        # Find closest quantile levels
        indices = [self._find_nearest_quantile_index(level) for level in q]

        # Build column names and data
        if len(self._targets) == 1:
            # Univariate
            columns = [f"q_{level:.3f}" for level in q]
            data = self._quantiles[:, indices]
        else:
            # Multivariate - quantile_matrix is (n_samples, n_targets * n_quantiles)
            # We need to extract per target
            columns = []
            data_parts = []

            for target_idx, target in enumerate(self._targets):
                for level in q:
                    columns.append(f"{target}_q_{level:.3f}")
                    q_idx = self._find_nearest_quantile_index(level)
                    data_parts.append(self._quantiles[:, target_idx * self._n_quantiles + q_idx])

            data = np.column_stack(data_parts)

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
            error_quantile_invalid(f"confidence must be in (0, 1), got {confidence}")

        alpha = (1 - confidence) / 2
        lower_idx = self._find_nearest_quantile_index(alpha)
        upper_idx = self._find_nearest_quantile_index(1 - alpha)

        if len(self._targets) == 1:
            # Univariate
            columns = ["lower", "upper"]
            data = np.column_stack(
                [
                    self._quantiles[:, lower_idx],
                    self._quantiles[:, upper_idx],
                ]
            )
        else:
            # Multivariate
            columns = []
            data_parts = []

            for target_idx, target in enumerate(self._targets):
                columns.append(f"{target}_lower")
                columns.append(f"{target}_upper")
                data_parts.append(self._quantiles[:, target_idx * self._n_quantiles + lower_idx])
                data_parts.append(self._quantiles[:, target_idx * self._n_quantiles + upper_idx])

            data = np.column_stack(data_parts)

        return pl.DataFrame(data, schema=columns, orient="row")

    def mean(self) -> pl.Series | pl.DataFrame:
        """
        Return the 0.5 quantile (median) as a point estimate.

        Returns:
            Polars Series for univariate, DataFrame for multivariate
        """
        median_idx = self._find_nearest_quantile_index(0.5)

        if len(self._targets) == 1:
            # Univariate
            return pl.Series("mean", self._quantiles[:, median_idx])
        else:
            # Multivariate
            data = np.column_stack(
                [
                    self._quantiles[:, target_idx * self._n_quantiles + median_idx]
                    for target_idx in range(len(self._targets))
                ]
            )
            return pl.DataFrame(data, schema=self._targets, orient="row")

    def median(self) -> pl.Series | pl.DataFrame:
        """Return the 0.5 quantile as a point estimate."""
        return self.mean()

    def sample(self, n: int, random_state: int | None = None) -> pl.DataFrame:
        """
        Draw n samples per input row via spline-interpolated inverse CDF.

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
        try:
            from scipy.interpolate import interp1d
        except ImportError:
            raise ImportError("scipy is required for sampling. Install with: pip install scipy")

        if not isinstance(n, int) or n < 1:
            raise InvalidDataError(f"n must be a positive integer, got {n}")

        total_samples = self._n_samples * n
        if total_samples > MAX_TOTAL_SAMPLES:
            raise InvalidDataError(
                f"Total samples ({total_samples:,}) exceeds maximum ({MAX_TOTAL_SAMPLES:,}). "
                f"Reduce n (currently {n}) or number of input rows ({self._n_samples})."
            )

        rng = np.random.default_rng(random_state)

        if n <= MAX_SAMPLE_CHUNK_SIZE:
            return self._sample_chunk(n, rng, interp1d)

        return self._sample_chunked(n, rng, interp1d)

    def _sample_chunk(
        self,
        n: int,
        rng: np.random.Generator,
        interp1d,
    ) -> pl.DataFrame:
        """Sample with a single chunk (n <= MAX_SAMPLE_CHUNK_SIZE)."""
        uniform_samples = rng.uniform(0, 1, size=(self._n_samples, n))
        uniform_clipped = np.clip(uniform_samples, self._levels[0], self._levels[-1])

        target_samples_list = []
        for target_idx in range(len(self._targets)):
            q_start = target_idx * self._n_quantiles
            q_end = q_start + self._n_quantiles
            quantile_values = self._quantiles[:, q_start:q_end]

            target_samples = self._vectorized_inverse_cdf(
                quantile_values, uniform_clipped, self._levels, interp1d
            )
            target_samples_list.append(target_samples.flatten())

        sample_matrix = np.column_stack(target_samples_list)
        sample_ids = np.repeat(np.arange(self._n_samples), n)

        result = pl.DataFrame(sample_matrix, schema=self._targets, orient="row")
        result.insert_column(0, pl.Series("sample_id", sample_ids))
        return result

    def _sample_chunked(
        self,
        n: int,
        rng: np.random.Generator,
        interp1d,
    ) -> pl.DataFrame:
        """Sample with chunking for large n values."""
        chunks = []
        remaining = n

        while remaining > 0:
            chunk_size = min(remaining, MAX_SAMPLE_CHUNK_SIZE)
            uniform_samples = rng.uniform(0, 1, size=(self._n_samples, chunk_size))
            uniform_clipped = np.clip(uniform_samples, self._levels[0], self._levels[-1])

            target_samples_list = []
            for target_idx in range(len(self._targets)):
                q_start = target_idx * self._n_quantiles
                q_end = q_start + self._n_quantiles
                quantile_values = self._quantiles[:, q_start:q_end]

                target_samples = self._vectorized_inverse_cdf(
                    quantile_values, uniform_clipped, self._levels, interp1d
                )
                target_samples_list.append(target_samples.flatten())

            sample_matrix = np.column_stack(target_samples_list)
            sample_ids = np.repeat(np.arange(self._n_samples), chunk_size)

            chunk_df = pl.DataFrame(sample_matrix, schema=self._targets, orient="row")
            chunk_df = chunk_df.with_columns(pl.Series("sample_id", sample_ids))
            cols = ["sample_id"] + self._targets
            chunk_df = chunk_df.select(cols)
            chunks.append(chunk_df)

            remaining -= chunk_size

        return pl.concat(chunks)

    @staticmethod
    def _vectorized_inverse_cdf(
        quantile_values: np.ndarray,
        uniform_clipped: np.ndarray,
        levels: np.ndarray,
        interp1d,
    ) -> np.ndarray:
        """
        Vectorized inverse CDF sampling.

        Uses vectorized operations where possible, falling back to row-wise
        interp1d only when necessary.

        Args:
            quantile_values: (n_samples, n_quantiles) array of quantile values
            uniform_clipped: (n_samples, n) array of uniform samples
            levels: (n_quantiles,) array of quantile levels
            interp1d: scipy.interpolate.interp1d constructor

        Returns:
            (n_samples, n) array of sampled values
        """
        n_samples = quantile_values.shape[0]
        n_draws = uniform_clipped.shape[1]
        result = np.zeros((n_samples, n_draws))

        for row_idx in range(n_samples):
            cdf_inverse = interp1d(
                levels,
                quantile_values[row_idx],
                kind="linear",
                fill_value="extrapolate",
            )
            result[row_idx] = cdf_inverse(uniform_clipped[row_idx])

        return result

    def plot(
        self,
        actuals: pl.Series | pl.DataFrame | None = None,
        confidence_bands: list[float] | None = None,
        title: str | None = None,
    ) -> None:
        """
        Fan chart of quantile bands.

        If actuals provided: overlays true values and computes empirical coverage.
        Requires matplotlib (soft dependency).

        Args:
            actuals: Optional actual values for comparison
            confidence_bands: Confidence levels to display (default: [0.5, 0.8, 0.9, 0.95])
            title: Optional plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        if confidence_bands is None:
            confidence_bands = [0.5, 0.8, 0.9, 0.95]

        if self._n_samples > PLOT_MAX_SAMPLES:
            # Downsample for plotting
            step = max(1, self._n_samples // PLOT_MAX_SAMPLES)
            plot_indices = slice(None, None, step)
        else:
            plot_indices = slice(None)

        fig, ax = plt.subplots(figsize=(12, 6))

        x_axis = range(self._n_samples)[plot_indices]

        # Plot confidence bands (darkest = narrowest)
        for confidence in reversed(confidence_bands):
            interval = self.interval(confidence)
            if len(self._targets) == 1:
                lower_col, upper_col = "lower", "upper"
            else:
                lower_col, upper_col = f"{self._targets[0]}_lower", f"{self._targets[0]}_upper"

            lower = interval[lower_col]
            upper = interval[upper_col]

            alpha = 0.1 + (1 - confidence) * 0.3
            ax.fill_between(
                x_axis,
                lower.to_numpy()[plot_indices],
                upper.to_numpy()[plot_indices],
                alpha=alpha,
                label=f"{confidence * 100:.0f}% interval",
            )

        # Plot median
        mean = self.mean()
        if isinstance(mean, pl.DataFrame):
            mean_series = mean[self._targets[0]]
        else:
            mean_series = mean

        ax.plot(mean_series.to_numpy()[plot_indices], label="Median", linewidth=2)

        # Plot actuals if provided
        if actuals is not None:
            if isinstance(actuals, pl.DataFrame):
                if len(self._targets) == 1:
                    actuals = actuals[self._targets[0]]
                else:
                    actuals = actuals[self._targets[0]]
            ax.plot(
                actuals.to_numpy()[plot_indices],
                label="Actuals",
                linewidth=1.5,
                alpha=0.7,
            )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        if title:
            ax.set_title(title)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    @lru_cache(maxsize=128)
    def _find_nearest_quantile_index(self, q: float) -> int:
        """Find index of nearest quantile level. Cached for repeated lookups."""
        distances = np.abs(self._levels - q)
        return int(np.argmin(distances))

    def __repr__(self) -> str:
        parts = [
            f"DistributionPrediction(n={self._n_samples}, "
            f"targets={self._targets}, quantiles={len(self._levels)}"
        ]
        if self._posterior is not None:
            parts.append(", posterior=True")
        parts.append(")")
        return "".join(parts)

    # --- Bayesian posterior methods ---

    def posterior_samples(self) -> np.ndarray:
        """Return raw MCMC posterior samples."""
        if self._posterior is None:
            error_invalid_data(
                "posterior_samples() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        assert self._posterior is not None
        return self._posterior

    def credible_interval(self, confidence: float = 0.9) -> pl.DataFrame:
        """Compute Bayesian credible interval from posterior."""
        if self._posterior is None:
            error_invalid_data(
                "credible_interval() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        assert self._posterior is not None
        if not (0 < confidence < 1):
            error_quantile_invalid(f"confidence must be in (0, 1), got {confidence}")
        alpha = (1 - confidence) / 2
        lower = np.quantile(self._posterior, alpha, axis=0)
        upper = np.quantile(self._posterior, 1 - alpha, axis=0)
        return pl.DataFrame({"lower": lower, "upper": upper}, orient="row")

    def rhat(self, n_chains: int = 4) -> np.ndarray:
        """Compute Gelman-Rubin R-hat convergence diagnostic."""
        if self._posterior is None:
            error_invalid_data(
                "rhat() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        assert self._posterior is not None
        samples = self._posterior
        n_total = samples.shape[0]
        chain_len = n_total // n_chains
        chains = samples[: n_chains * chain_len].reshape(n_chains, chain_len, -1)
        chain_means = chains.mean(axis=1)
        b = chain_len * np.var(chain_means, axis=0, ddof=1)
        w = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
        var_hat = (1 - 1 / chain_len) * w + (1 / chain_len) * b
        return np.sqrt(var_hat / (w + 1e-10))  # type: ignore

    def posterior_summary(self) -> pl.DataFrame:
        """Return summary statistics of posterior samples."""
        if self._posterior is None:
            error_invalid_data(
                "posterior_summary() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        assert self._posterior is not None
        return pl.DataFrame(
            {
                "param": [f"param_{i}" for i in range(self._posterior.shape[1])],
                "mean": np.mean(self._posterior, axis=0),
                "std": np.std(self._posterior, axis=0),
                "q025": np.quantile(self._posterior, 0.025, axis=0),
                "q50": np.quantile(self._posterior, 0.5, axis=0),
                "q975": np.quantile(self._posterior, 0.975, axis=0),
            }
        )

    # --- Multi-modal group methods ---

    def group_uncertainty(self) -> dict[str, float]:
        """Return per-group uncertainty contribution (interval width)."""
        if self._group_predictions is None:
            error_invalid_data(
                "group_uncertainty() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        assert self._group_predictions is not None
        result = {}
        for name, pred in self._group_predictions.items():
            interval = pred.interval(0.9)
            width = (interval["upper"] - interval["lower"]).mean()
            result[name] = float(width)  # type: ignore[arg-type]
        return result

    def group_intervals(self, confidence: float = 0.9) -> dict[str, pl.DataFrame]:
        """Return per-group prediction intervals."""
        if self._group_predictions is None:
            error_invalid_data(
                "group_intervals() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        assert self._group_predictions is not None
        return {name: pred.interval(confidence) for name, pred in self._group_predictions.items()}

    def cross_group_correlation(self) -> np.ndarray:
        """Return cross-group correlation matrix based on group median predictions."""
        if self._group_predictions is None:
            error_invalid_data(
                "cross_group_correlation() requires group predictions. "
                "Use a CrossModalAggregator to generate predictions with groups."
            )
        assert self._group_predictions is not None
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
            error_invalid_data(
                "treatment_effect() requires treatment info. "
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        assert self._treatment_info is not None
        return self._treatment_info["cate"]  # type: ignore

    def average_treatment_effect(self) -> dict:
        """Return ATE with confidence interval."""
        if self._treatment_info is None:
            error_invalid_data(
                "average_treatment_effect() requires treatment info. "
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        assert self._treatment_info is not None
        return {
            "ate": self._treatment_info["ate"],
            "ci": self._treatment_info["ate_ci"],
        }

    def heterogeneity_score(self) -> float:
        """Return CATE variance as heterogeneity measure."""
        if self._treatment_info is None:
            error_invalid_data(
                "heterogeneity_score() requires treatment info. "
                "Use a CausalUncertaintyEstimator to generate predictions with treatment data."
            )
        assert self._treatment_info is not None
        return float(np.var(self._treatment_info["cate"]))
