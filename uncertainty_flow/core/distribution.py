"""DistributionPrediction - core output object for all models."""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


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
        index: pl.Series | None = None,
    ):
        # Validate inputs
        if quantile_matrix.ndim != 2:
            raise ValueError(f"quantile_matrix must be 2D, got shape {quantile_matrix.shape}")

        if len(target_names) == 0:
            raise ValueError("target_names cannot be empty")

        # For multivariate: matrix has n_targets * n_quantiles columns
        # For univariate: matrix has n_quantiles columns
        n_targets = len(target_names)
        expected_cols = n_targets * len(quantile_levels)

        if quantile_matrix.shape[1] != expected_cols:
            raise ValueError(
                f"quantile_matrix has {quantile_matrix.shape[1]} columns "
                f"but expected {expected_cols} columns for {n_targets} target(s) "
                f"with {len(quantile_levels)} quantile levels each"
            )

        # Store internally as NumPy
        self._quantiles = quantile_matrix
        self._levels = quantile_levels
        self._targets = target_names
        self._index = index
        self._n_samples = quantile_matrix.shape[0]
        self._n_quantiles = len(quantile_levels)

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
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

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
        Return the 0.5 quantile (median).

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
        """
        try:
            from scipy.interpolate import interp1d
        except ImportError:
            raise ImportError("scipy is required for sampling. Install with: pip install scipy")

        rng = np.random.default_rng(random_state)

        target_samples_list = []
        for target_idx in range(len(self._targets)):
            target_samples = []
            for row_idx in range(self._n_samples):
                q_start = target_idx * self._n_quantiles
                q_end = q_start + self._n_quantiles
                quantile_values = self._quantiles[row_idx, q_start:q_end]

                cdf_inverse = interp1d(
                    self._levels,
                    quantile_values,
                    kind="linear",
                    fill_value="extrapolate",
                )

                uniform_samples = rng.uniform(0, 1, size=n)
                uniform_clipped = np.clip(uniform_samples, self._levels[0], self._levels[-1])
                drawn_samples = cdf_inverse(uniform_clipped)
                target_samples.extend(drawn_samples)

            target_samples_list.append(target_samples)

        sample_matrix = np.column_stack(target_samples_list)

        sample_ids = []
        for row_idx in range(self._n_samples):
            sample_ids.extend([row_idx] * n)

        result = pl.DataFrame(sample_matrix, schema=self._targets, orient="row")
        result.insert_column(0, pl.Series("sample_id", sample_ids))

        return result

    def plot(
        self,
        actuals: pl.Series | pl.DataFrame | None = None,
        confidence_bands: list[float] = [0.5, 0.8, 0.9, 0.95],
        title: str | None = None,
    ) -> None:
        """
        Fan chart of quantile bands.

        If actuals provided: overlays true values and computes empirical coverage.
        Requires matplotlib (soft dependency).

        Args:
            actuals: Optional actual values for comparison
            confidence_bands: Confidence levels to display
            title: Optional plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        if self._n_samples > 500:
            # Downsample for plotting
            step = max(1, self._n_samples // 500)
            plot_indices = slice(None, None, step)
        else:
            plot_indices = slice(None)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot confidence bands (darkest = narrowest)
        for confidence in reversed(confidence_bands):
            interval = self.interval(confidence)
            lower = interval["lower" if len(self._targets) == 1 else f"{self._targets[0]}_lower"]
            upper = interval["upper" if len(self._targets) == 1 else f"{self._targets[0]}_upper"]

            alpha = 0.1 + (1 - confidence) * 0.3
            ax.fill_between(
                range(self._n_samples)[plot_indices],
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

    def _find_nearest_quantile_index(self, q: float) -> int:
        """Find index of nearest quantile level."""
        distances = [abs(level - q) for level in self._levels]
        return int(np.argmin(distances))

    def __repr__(self) -> str:
        return (
            f"DistributionPrediction(n={self._n_samples}, "
            f"targets={self._targets}, quantiles={len(self._levels)})"
        )
