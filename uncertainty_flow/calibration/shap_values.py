"""SHAP-based uncertainty feature attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


def uncertainty_shap(
    model,
    X: pl.DataFrame,  # noqa: N803
    background: pl.DataFrame | None = None,
    quantile_pairs: list[tuple[float, float]] | None = None,
) -> pl.DataFrame:
    """
    Compute SHAP values for quantile intervals to identify interval width drivers.

    Runs SHAP on the upper and lower quantile predictions separately,
    then computes the difference to identify what drives interval width.

    Args:
        model: Fitted uncertainty model with predict() returning DistributionPrediction
        X: Feature DataFrame for SHAP evaluation
        background: Background dataset for SHAP explanation.
                   If None, uses X[:100] as suggested in roadmap.
        quantile_pairs: List of (lower, upper) quantile pairs to analyze.
                       Defaults to [(0.1, 0.9), (0.05, 0.95)].

    Returns:
        Polars DataFrame with columns:
        - feature: Feature name
        - quantile_level: e.g., "Q0.1", "Q0.9"
        - shap_value: Mean absolute SHAP value
        - interval_width_contribution: Difference between upper and lower SHAP

    Examples:
        >>> import numpy as np
        >>> import polars as pl
        >>> from uncertainty_flow import DeepQuantileNet
        >>> from uncertainty_flow.calibration import uncertainty_shap
        >>>
        >>> np.random.seed(42)
        >>> n = 100
        >>> df = pl.DataFrame({
        ...     "x1": np.random.randn(n),
        ...     "x2": np.random.randn(n),
        ...     "y": 3 * np.random.randn(n) + 5,
        ... })
        >>> model = DeepQuantileNet(random_state=42)
        >>> model.fit(df, target="y")
        >>> X = df.select(["x1", "x2"])
        >>> shap_df = uncertainty_shap(model, X)
        >>> shap_df
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap is required for uncertainty_shap. "
            "Install with: pip install 'uncertainty-flow[shap]'"
        )

    if quantile_pairs is None:
        quantile_pairs = [(0.1, 0.9), (0.05, 0.95)]

    if background is None:
        background = X.head(100)

    background_np = background.to_numpy()
    x_np = X.to_numpy()
    feature_names = X.columns

    pred = model.predict(X.head(1))
    n_targets = len(pred.target_names)

    results = []

    for target_idx, target_name in enumerate(pred.target_names):
        for lower_q, upper_q in quantile_pairs:

            def lower_quantile_model(x):
                preds = model.predict(pl.DataFrame(x, schema=X.columns))
                q_matrix = preds._quantiles
                lower_idx = preds._find_nearest_quantile_index(lower_q)

                if n_targets == 1:
                    return q_matrix[:, lower_idx]
                else:
                    start = target_idx * preds._n_quantiles
                    return q_matrix[:, start + lower_idx]

            def upper_quantile_model(x):
                preds = model.predict(pl.DataFrame(x, schema=X.columns))
                q_matrix = preds._quantiles
                upper_idx = preds._find_nearest_quantile_index(upper_q)

                if n_targets == 1:
                    return q_matrix[:, upper_idx]
                else:
                    start = target_idx * preds._n_quantiles
                    return q_matrix[:, start + upper_idx]

            try:
                lower_explainer = shap.KernelExplainer(lower_quantile_model, background_np)
                lower_shap = lower_explainer.shap_values(x_np)

                upper_explainer = shap.KernelExplainer(upper_quantile_model, background_np)
                upper_shap = upper_explainer.shap_values(x_np)

                for feat_idx, feat_name in enumerate(feature_names):
                    lower_vals = lower_shap[:, feat_idx]
                    upper_vals = upper_shap[:, feat_idx]

                    results.append(
                        {
                            "feature": feat_name,
                            "target": target_name,
                            f"shap_Q{lower_q}": float(np.mean(np.abs(lower_vals))),
                            f"shap_Q{upper_q}": float(np.mean(np.abs(upper_vals))),
                            "interval_width_contribution": float(
                                np.mean(np.abs(upper_vals - lower_vals))
                            ),
                        }
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

    if not results:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "target": pl.String,
                "shap_Q0.1": pl.Float64,
                "shap_Q0.9": pl.Float64,
                "interval_width_contribution": pl.Float64,
            }
        )

    return pl.DataFrame(results)
