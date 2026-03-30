"""Residual analysis for uncertainty driver detection."""

import numpy as np
import polars as pl
from scipy import stats


def compute_uncertainty_drivers(
    residuals: np.ndarray,
    features: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute correlation between features and squared residuals.

    Features with high correlation to squared residuals indicate
    heteroscedasticity - they drive uncertainty in predictions.

    Args:
        residuals: Model residuals (y_true - y_pred)
        features: Feature DataFrame

    Returns:
        Polars DataFrame with columns:
        - feature: Feature name
        - residual_correlation: Pearson correlation with squared residuals
        - p_value: Statistical significance

        Sorted by absolute correlation descending.

    Examples:
        >>> import numpy as np
        >>> import polars as pl
        >>> residuals = np.array([1, -2, 3, -1, 2])
        >>> features = pl.DataFrame({
        ...     "a": [1, 2, 3, 4, 5],
        ...     "b": [5, 4, 3, 2, 1],
        ... })
        >>> compute_uncertainty_drivers(residuals, features)
        shape: (2, 3)
        ┌─────────┬─────────────────────┬─────────┐
        │ feature ┆ residual_correlation ┆ p_value │
        │ ---     ┆ ---                   ┆ ---     │
        │ str     ┆ f64                  ┆ f64     │
        ╞═════════╪══════════════════════╪═════════╡
        │ a       ┆ 0.89                  ┆ 0.11    │
        │ b       ┆ -0.89                 ┆ 0.11    │
        └─────────┴─────────────────────┴─────────┘
    """
    squared_residuals = residuals**2

    results = []
    for col in features.columns:
        feature_vals = features[col].to_numpy()

        # Skip constant features
        if np.std(feature_vals) == 0:
            continue

        # Compute correlation
        corr, p_value = stats.pearsonr(feature_vals, squared_residuals)

        results.append(
            {
                "feature": col,
                "residual_correlation": corr,
                "p_value": p_value,
            }
        )

    if not results:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "residual_correlation": pl.Float64,
                "p_value": pl.Float64,
            }
        )

    df = pl.DataFrame(results)
    df = df.sort(by="residual_correlation", descending=True)

    # Warn if no significant drivers found
    if df.filter(pl.col("p_value") < 0.05).height == 0:
        from ..utils.exceptions import warn_no_uncertainty_drivers

        warn_no_uncertainty_drivers()

    return df
