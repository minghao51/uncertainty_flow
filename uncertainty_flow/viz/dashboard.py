"""Interactive Streamlit dashboard for uncertainty exploration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..utils.polars_bridge import to_numpy_series_zero_copy

if TYPE_CHECKING:
    from ..core.base import BaseUncertaintyModel


def launch_dashboard(
    model: "BaseUncertaintyModel",
    data: pl.DataFrame,
    target: str | None = None,
    port: int = 8050,
    title: str | None = None,
) -> None:
    """
    Launch interactive Streamlit dashboard for uncertainty exploration.

    Provides visualizations for calibration, prediction intervals,
    residual analysis, feature-uncertainty relationships, and more.

    Parameters
    ----------
    model : BaseUncertaintyModel
        Fitted uncertainty model with predict() method
    data : pl.DataFrame
        Dataset for analysis (features + target)
    target : str, optional
        Target column name. If None, will be inferred from model
    port : int, default=8050
        Port for Streamlit server
    title : str, optional
        Dashboard title. If None, uses model name

    Examples
    --------
    >>> import polars as pl
    >>> from uncertainty_flow.models import QuantileForestForecaster
    >>> from uncertainty_flow.viz import launch_dashboard
    >>>
    >>> model = QuantileForestForecaster(targets="demand", horizon=7)
    >>> model.fit(train_data)
    >>>
    >>> # Launch dashboard
    >>> launch_dashboard(model, test_data, port=8050)

    Notes
    -----
    Requires streamlit as an optional dependency:
        pip install uncertainty-flow[viz]

    The dashboard includes:
        - Calibration curves (coverage vs. confidence)
        - Interval width distribution
        - Residual analysis plots
        - Feature-uncertainty relationships
        - Time series fan charts
        - Leverage analysis reports
    """
    if model is None:
        raise ValueError("model is required")
    if data.is_empty():
        raise ValueError("data must contain at least one row")

    try:
        import streamlit as st
    except ImportError:
        raise ImportError(
            "streamlit is required for the dashboard. Install with: pip install streamlit"
        )

    # Set page config
    if title is None:
        title = f"Uncertainty Analysis Dashboard - {model.__class__.__name__}"

    st.set_page_config(
        page_title="Uncertainty Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(title)

    # Sidebar controls
    st.sidebar.header("Configuration")

    # Confidence level slider
    confidence = st.sidebar.slider(
        "Confidence Level",
        min_value=0.5,
        max_value=0.99,
        value=0.9,
        step=0.01,
        help="Confidence level for prediction intervals",
    )

    # Sample size for plots
    max_samples = st.sidebar.slider(
        "Max Samples for Plots",
        min_value=50,
        max_value=5000,
        value=1000,
        step=50,
        help="Maximum number of samples to display in plots",
    )

    # Infer target if not provided
    if target is None and hasattr(model, "_targets"):
        targets = model._targets
        if targets and len(targets) == 1:
            target = targets[0]

    if target is None:
        st.error("Could not infer target column. Please specify 'target' parameter.")
        return

    # Separate features and target
    feature_cols = [col for col in data.columns if col != target]
    features = data.select(feature_cols)
    y_true = data[target]

    # Downsample if needed
    if len(features) > max_samples:
        indices = np.linspace(0, len(features) - 1, max_samples, dtype=int)
        features = features[indices]
        y_true = y_true[indices]

    # Generate predictions
    with st.spinner("Generating predictions..."):
        predictions = model.predict(features)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Calibration", "Intervals", "Residuals", "Feature Analysis", "Summary"]
    )

    with tab1:
        _render_calibration_tab(predictions, y_true, confidence)

    with tab2:
        _render_intervals_tab(predictions, y_true, confidence)

    with tab3:
        _render_residuals_tab(predictions, y_true, features, confidence)

    with tab4:
        _render_feature_analysis_tab(predictions, y_true, features, confidence)

    with tab5:
        _render_summary_tab(model, predictions, y_true, confidence)


def _render_calibration_tab(predictions, y_true, confidence):
    """Render calibration analysis tab."""
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("Calibration Analysis")

    # Compute empirical coverage for multiple confidence levels
    confidence_levels = np.linspace(0.5, 0.95, 10)
    empirical_coverages = []

    for level in confidence_levels:
        interval = predictions.interval(level)
        if len(predictions._targets) == 1:
            lower = interval["lower"].to_numpy()
            upper = interval["upper"].to_numpy()
        else:
            first_target = predictions._targets[0]
            lower = interval[f"{first_target}_lower"].to_numpy()
            upper = interval[f"{first_target}_upper"].to_numpy()

        y_true_arr = to_numpy_series_zero_copy(y_true)
        coverage = np.mean((y_true_arr >= lower) & (y_true_arr <= upper))
        empirical_coverages.append(coverage)

    # Plot calibration curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(confidence_levels, empirical_coverages, "bo-", label="Empirical", linewidth=2)
    ax.plot(confidence_levels, confidence_levels, "r--", label="Ideal", linewidth=2)
    ax.axvline(
        confidence, color="gray", linestyle=":", alpha=0.5, label=f"Current ({confidence:.2f})"
    )
    ax.axhline(
        empirical_coverages[np.argmin(np.abs(confidence_levels - confidence))],
        color="gray",
        linestyle=":",
        alpha=0.5,
    )
    ax.set_xlabel("Target Confidence Level")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 0.95)
    ax.set_ylim(0.5, 1.0)

    st.pyplot(fig)

    # Coverage metrics
    current_coverage = empirical_coverages[np.argmin(np.abs(confidence_levels - confidence))]
    coverage_error = abs(current_coverage - confidence)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Coverage", f"{confidence:.1%}")
    with col2:
        st.metric("Empirical Coverage", f"{current_coverage:.1%}")
    with col3:
        st.metric("Coverage Error", f"{coverage_error:.1%}")


def _render_intervals_tab(predictions, y_true, confidence):
    """Render interval analysis tab."""
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("Prediction Intervals")

    interval = predictions.interval(confidence)

    if len(predictions._targets) == 1:
        lower = interval["lower"].to_numpy()
        upper = interval["upper"].to_numpy()
    else:
        first_target = predictions._targets[0]
        lower = interval[f"{first_target}_lower"].to_numpy()
        upper = interval[f"{first_target}_upper"].to_numpy()

    widths = upper - lower

    # Plot interval width distribution
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(widths, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(
            np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.2f}"
        )
        ax.set_xlabel("Interval Width")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Prediction Interval Widths")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(widths, vert=False)
        ax.set_xlabel("Interval Width")
        ax.set_title("Interval Width Box Plot")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)

    # Metrics
    st.subheader("Interval Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Width", f"{np.mean(widths):.2f}")
    with col2:
        st.metric("Median Width", f"{np.median(widths):.2f}")
    with col3:
        st.metric("Std Width", f"{np.std(widths):.2f}")
    with col4:
        st.metric("Max Width", f"{np.max(widths):.2f}")


def _render_residuals_tab(predictions, y_true, features, confidence):
    """Render residual analysis tab."""
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("Residual Analysis")

    interval = predictions.interval(confidence)

    if len(predictions._targets) == 1:
        lower = interval["lower"].to_numpy()
        upper = interval["upper"].to_numpy()
        mean = predictions.mean().to_numpy()
    else:
        first_target = predictions._targets[0]
        lower = interval[f"{first_target}_lower"].to_numpy()
        upper = interval[f"{first_target}_upper"].to_numpy()
        mean = predictions.mean().select(first_target).to_numpy().flatten()

    y_true_arr = to_numpy_series_zero_copy(y_true)

    # Compute residuals
    residuals = y_true_arr - mean

    # Plot residual distribution
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="Zero")
        ax.axvline(
            np.mean(residuals),
            color="green",
            linestyle="--",
            label=f"Mean: {np.mean(residuals):.2f}",
        )
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mean, residuals, alpha=0.5, s=10)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Predicted")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Coverage by predicted value
    st.subheader("Coverage by Predicted Quantile")

    n_bins = 5
    quantile_bins = pdcut(mean, n_bins)
    bin_coverage = []

    for i in range(n_bins):
        mask = quantile_bins == i
        if mask.sum() > 0:
            bin_lower = lower[mask]
            bin_upper = upper[mask]
            bin_y_true = y_true_arr[mask]
            coverage = np.mean((bin_y_true >= bin_lower) & (bin_y_true <= bin_upper))
            bin_coverage.append(coverage)
        else:
            bin_coverage.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(n_bins), bin_coverage, alpha=0.7, edgecolor="black")
    ax.axhline(confidence, color="red", linestyle="--", label=f"Target: {confidence:.2f}")
    ax.set_xlabel("Predicted Value Quantile Bin")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Coverage by Predicted Quantile")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig)


def _render_feature_analysis_tab(predictions, y_true, features, confidence):
    """Render feature-uncertainty relationship tab."""
    import matplotlib.pyplot as plt
    import streamlit as st

    st.subheader("Feature-Uncertainty Relationships")

    interval = predictions.interval(confidence)

    if len(predictions._targets) == 1:
        lower = interval["lower"].to_numpy()
        upper = interval["upper"].to_numpy()
    else:
        first_target = predictions._targets[0]
        lower = interval[f"{first_target}_lower"].to_numpy()
        upper = interval[f"{first_target}_upper"].to_numpy()

    widths = upper - lower

    # Feature selection
    feature_cols = features.columns[:6]  # Limit to first 6 features
    selected_feature = st.selectbox("Select Feature to Analyze", feature_cols)

    if selected_feature:
        feature_vals = features[selected_feature].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(feature_vals, widths, alpha=0.5, s=10, c=widths, cmap="viridis")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Interval Width")
        ax.set_title(f"Interval Width vs {selected_feature}")
        plt.colorbar(scatter, ax=ax, label="Width")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Correlation
        corr = np.corrcoef(feature_vals, widths)[0, 1]
        st.metric(f"Correlation ({selected_feature})", f"{corr:.3f}")


def _render_summary_tab(model, predictions, y_true, confidence):
    """Render summary tab."""
    import streamlit as st

    st.subheader("Model Summary")

    # Model info
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Configuration**")
        st.write(f"Type: `{model.__class__.__name__}`")
        if hasattr(model, "summary"):
            st.write(model.summary())

    with col2:
        st.write("**Prediction Summary**")
        st.write(f"Number of samples: `{len(y_true)}`")
        st.write(f"Confidence level: `{confidence:.1%}`")

        interval = predictions.interval(confidence)
        if len(predictions._targets) == 1:
            lower = interval["lower"].to_numpy()
            upper = interval["upper"].to_numpy()
        else:
            first_target = predictions._targets[0]
            lower = interval[f"{first_target}_lower"].to_numpy()
            upper = interval[f"{first_target}_upper"].to_numpy()

        y_true_arr = to_numpy_series_zero_copy(y_true)
        coverage = np.mean((y_true_arr >= lower) & (y_true_arr <= upper))
        widths = upper - lower

        st.write(f"Empirical coverage: `{coverage:.1%}`")
        st.write(f"Mean interval width: `{np.mean(widths):.2f}`")


def pdcut(x, n_bins):
    """Simple quantile binning for visualization."""
    x = np.asarray(x)
    if x.size == 0:
        return np.array([], dtype=int)

    if np.allclose(x, x[0]):
        return np.zeros_like(x, dtype=int)

    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(x, quantiles)
    bins = np.unique(bins)

    if bins.size <= 2:
        return np.zeros_like(x, dtype=int)

    return np.digitize(x, bins[1:-1], right=False)
