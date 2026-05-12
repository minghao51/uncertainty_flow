"""Plotting functions for DistributionPrediction (requires matplotlib)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..utils.polars_bridge import to_numpy_series

if TYPE_CHECKING:
    from ..core.distribution import DistributionPrediction

PLOT_MAX_SAMPLES = 500


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )


def _plot_slice(n_samples: int) -> slice:
    if n_samples > PLOT_MAX_SAMPLES:
        step = max(1, n_samples // PLOT_MAX_SAMPLES)
        return slice(None, None, step)
    return slice(None)


def _resolve_targets(
    pred: DistributionPrediction, targets: str | list[str], max_targets: int
) -> list[str]:
    if isinstance(targets, str):
        if targets == "all":
            selected = list(pred._targets)
        elif targets in pred._targets:
            selected = [targets]
        else:
            raise ValueError(f"Target {targets!r} not found. Available: {pred._targets}")
    else:
        selected = list(targets)

    for t in selected:
        if t not in pred._targets:
            raise ValueError(f"Target {t!r} not found. Available: {pred._targets}")

    if len(selected) > max_targets:
        warnings.warn(
            f"{len(selected)} targets requested but max_targets={max_targets}. "
            f"Showing first {max_targets}.",
            UserWarning,
            stacklevel=3,
        )
        selected = selected[:max_targets]

    return selected


def _subplot_grid(n_panels: int, n_cols: int = 3):
    """Create a matplotlib subplot grid, returning (fig, flat_axes_array)."""
    import matplotlib.pyplot as plt

    n_cols = min(n_cols, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes_flat = np.asarray(axes).flatten()
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)
    return fig, axes_flat


def _plot_quantile_fan(
    ax,
    pred: DistributionPrediction,
    target: str,
    actuals: pl.Series | pl.DataFrame | np.ndarray | None,
    confidence_bands: list[float],
    plot_indices: slice,
    title: str | None = None,
) -> None:
    x_axis = range(pred._n_samples)[plot_indices]

    for confidence in reversed(confidence_bands):
        interval = pred.interval(confidence)
        lower_col, upper_col = pred._interval_columns(target)

        lower = interval[lower_col]
        upper = interval[upper_col]

        alpha = 0.1 + (1 - confidence) * 0.3
        ax.fill_between(
            x_axis,
            to_numpy_series(lower)[plot_indices],
            to_numpy_series(upper)[plot_indices],
            alpha=alpha,
            label=f"{confidence * 100:.0f}% interval",
        )

    median_df = pred.median()
    if isinstance(median_df, pl.DataFrame):
        median_series = median_df[target]
    else:
        median_series = median_df
    ax.plot(to_numpy_series(median_series)[plot_indices], label="Median", linewidth=2)

    if actuals is not None:
        if isinstance(actuals, pl.DataFrame):
            actuals_series = actuals[target]
        elif isinstance(actuals, pl.Series):
            actuals_series = actuals
        else:
            actuals_series = actuals
        ax.plot(
            to_numpy_series(actuals_series)[plot_indices],
            label="Actuals",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")


def _plot_pit_bar(ax, hist_df: pl.DataFrame, title: str | None = None) -> None:
    centers = hist_df["bin_center"].to_numpy()
    counts = hist_df["count"].to_numpy()
    expected = hist_df["expected"].to_numpy()
    width = centers[1] - centers[0] if len(centers) > 1 else 0.1

    ax.bar(centers, counts, width=width * 0.9, alpha=0.7, label="PIT counts")
    ax.axhline(y=expected[0], color="red", linestyle="--", label="Uniform reference")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Count")
    ax.set_title(title or "PIT Histogram")
    ax.legend(loc="best")


def plot(
    pred: DistributionPrediction,
    actuals: pl.Series | pl.DataFrame | None = None,
    confidence_bands: list[float] | None = None,
    title: str | None = None,
    targets: str | list[str] = "all",
    max_targets: int = 6,
) -> None:
    """Fan chart of quantile bands. Requires matplotlib."""
    _require_mpl()
    import matplotlib.pyplot as plt

    if confidence_bands is None:
        confidence_bands = [0.5, 0.8, 0.9, 0.95]

    target_list = _resolve_targets(pred, targets, max_targets)
    plot_indices = _plot_slice(pred._n_samples)

    if len(target_list) == 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_quantile_fan(ax, pred, target_list[0], actuals, confidence_bands, plot_indices, title)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes_flat = _subplot_grid(len(target_list))
        for i, target in enumerate(target_list):
            _plot_quantile_fan(
                axes_flat[i], pred, target, actuals, confidence_bands, plot_indices, target
            )
        fig.suptitle(title or "")
        plt.tight_layout()
        plt.show()


def plot_pit(
    pred: DistributionPrediction,
    y_true: pl.Series | pl.DataFrame | np.ndarray,
    n_bins: int = 10,
) -> None:
    """Plot PIT histogram with uniform reference line. Requires matplotlib."""
    _require_mpl()
    import matplotlib.pyplot as plt

    hist = pred.pit_histogram(y_true, n_bins=n_bins)

    if isinstance(hist, dict):
        targets = list(hist.keys())
        n_targets = len(targets)
        fig, axes_flat = _subplot_grid(n_targets)
        for i, t in enumerate(targets):
            _plot_pit_bar(axes_flat[i], hist[t], title=t)
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_pit_bar(ax, hist)
        plt.tight_layout()
        plt.show()
