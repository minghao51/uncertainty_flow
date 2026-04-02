# API_SPEC.md — Full API Specification

---

## Overview

All public classes follow a consistent `fit` / `predict` interface. Inputs are Polars DataFrames or LazyFrames. Outputs are `DistributionPrediction` objects or Polars DataFrames.

---

## 1. `ConformalRegressor`

> Tabular regression. Wraps any scikit-learn estimator with statistically guaranteed coverage intervals.  
> **Coverage guarantee: ✅ (exchangeability assumption)**  
> **Non-crossing: ✅ (post-sort)**

```python
class ConformalRegressor(BaseUncertaintyModel):

    def __init__(
        self,
        base_model,                          # Any sklearn-compatible estimator
        calibration_method: str = "holdout", # "holdout" | "cross"
        calibration_size: float = 0.2,       # Fraction for holdout
        coverage_target: float = 0.9,        # Default interval width
        auto_tune: bool = True,              # Tune supported params before final fit
        uncertainty_features: list[str] | None = None,  # User hint for heteroscedasticity
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target: str,
    ) -> "ConformalRegressor":
        """
        Fits base model on training portion.
        Runs residual correlation analysis post-fit → populates uncertainty_drivers_.
        Warns if calibration set < 50 samples.
        Raises if calibration set < 20 samples.
        """
        ...

    def predict(
        self,
        data: pl.DataFrame | pl.LazyFrame,
    ) -> DistributionPrediction:
        """
        Returns DistributionPrediction with quantile levels:
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        """
        ...

    def calibration_report(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target: str,
        quantile_levels: list[float] | None = None,  # Defaults to [0.8, 0.9, 0.95]
    ) -> pl.DataFrame:
        """
        Returns Polars DataFrame schema:
        ┌────────────┬──────────────────┬───────────────────┬──────────┬───────────────┐
        │ quantile   │ requested_coverage│ achieved_coverage │ sharpness│ winkler_score │
        │ f64        │ f64               │ f64               │ f64      │ f64           │
        └────────────┴──────────────────┴───────────────────┴──────────┴───────────────┘
        Also emits warning rows where |requested - achieved| > 0.05.
        """
        ...

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """
        Set after .fit(). Schema:
        ┌─────────────┬─────────────────────┬─────────┐
        │ feature     │ residual_correlation │ p_value │
        │ str         │ f64                  │ f64     │
        └─────────────┴─────────────────────┴─────────┘
        Sorted descending by |residual_correlation|.
        None if not yet fitted.
        """
        ...
```

---

## 2. `ConformalForecaster`

> Time series forecasting (univariate & multivariate). Temporal-aware conformal wrapper.  
> **Coverage guarantee: ✅ (with temporal correction)**  
> **Non-crossing: ✅ (post-sort)**

```python
class ConformalForecaster(BaseUncertaintyModel):

    def __init__(
        self,
        base_model,                          # Any sklearn-compatible estimator
        horizon: int,                        # Forecast horizon (steps ahead)
        targets: str | list[str],            # Single or multiple target columns
        target_correlation: str = "auto",    # "auto" | "independent"
                                             # "auto" fits Gaussian copula
        lags: int | list[int] = 1,           # Lag features auto-generated
        calibration_method: str = "holdout", # "holdout" | "cross"
        calibration_size: float = 0.2,       # Always takes LAST n% (temporal)
        auto_tune: bool = True,              # Tune supported params before final fit
        uncertainty_features: list[str] | None = None,
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
    ) -> "ConformalForecaster":
        """
        Temporal holdout is always from the END of the series.
        Fits Gaussian copula on residuals if target_correlation='auto' and
        len(targets) > 1.
        """
        ...

    def predict(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        steps: int | None = None,   # Defaults to self.horizon
    ) -> DistributionPrediction:
        ...

    def calibration_report(self, ...) -> pl.DataFrame: ...

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None: ...
```

---

## 3. `QuantileForestForecaster`

> Quantile Regression Forest. Stores full leaf distributions for fast quantile retrieval.  
> **Coverage guarantee: ⚠️ Empirical only**  
> **Non-crossing: ✅ (by leaf distribution construction)**

```python
class QuantileForestForecaster(BaseUncertaintyModel):

    def __init__(
        self,
        targets: str | list[str],
        horizon: int,
        n_estimators: int = 200,
        min_samples_leaf: int = 5,            # Controls distribution richness per leaf
        target_correlation: str = "auto",
        calibration_size: float = 0.2,
        auto_tune: bool = True,
        uncertainty_features: list[str] | None = None,
        random_state: int | None = None,
    ): ...

    def fit(self, data: pl.DataFrame | pl.LazyFrame) -> "QuantileForestForecaster": ...

    def predict(self, data: pl.DataFrame | pl.LazyFrame) -> DistributionPrediction: ...

    def calibration_report(self, ...) -> pl.DataFrame: ...

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None: ...
```

---

## 4. `DistributionPrediction`

> Core output object. Returned by all `.predict()` calls.

```python
class DistributionPrediction:

    def quantile(
        self,
        q: float | list[float],
    ) -> pl.DataFrame:
        """
        Returns Polars DataFrame.
        Single float → one column named f"q_{q}".
        List → multiple columns named f"q_{qi}" for each qi.
        For multivariate, columns are named f"{target}_q_{qi}".
        """
        ...

    def interval(
        self,
        confidence: float = 0.9,
    ) -> pl.DataFrame:
        """
        Returns Polars DataFrame with columns: lower, upper.
        Derives quantiles: alpha = (1 - confidence) / 2
        lower = quantile(alpha), upper = quantile(1 - alpha)
        For multivariate: {target}_lower, {target}_upper per target.
        """
        ...

    def mean(self) -> pl.Series | pl.DataFrame:
        """
        Returns the 0.5 quantile (median).
        Series for univariate, DataFrame for multivariate.
        """
        ...

    def sample(
        self,
        n: int,
        random_state: int | None = None,
    ) -> pl.DataFrame:
        """
        Draw n samples per input row via spline-interpolated inverse CDF.
        Uses scipy.interpolate.interp1d to build an inverse CDF from quantile matrix,
        then draws uniform samples and maps through inverse CDF.
        Returns Polars DataFrame with (n * n_samples) rows and columns: sample_id, plus one column per target.
        sample_id: index of original input row (0 to n_samples-1, repeated n times).
        """
        ...

    def plot(
        self,
        actuals: pl.Series | pl.DataFrame | None = None,
        confidence_bands: list[float] = [0.5, 0.8, 0.9, 0.95],
        title: str | None = None,
    ) -> None:
        """
        Fan chart of quantile bands (darkest = narrowest interval).
        If actuals provided: overlays true values and computes empirical coverage.
        Requires matplotlib (soft dependency).
        """
        ...

    def __repr__(self) -> str:
        """
        Example:
        DistributionPrediction(n=500, targets=['price'], quantiles=11, coverage_target=0.90)
        """
        ...
```

---

## 5. Metrics (standalone, importable independently)

```python
from uncertainty_flow.metrics import pinball_loss, winkler_score, coverage_score

# Pinball loss (quantile loss)
pinball_loss(
    y_true: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
    quantile: float,                  # e.g. 0.9
) -> float

# Winkler interval score
winkler_score(
    y_true: pl.Series | np.ndarray,
    lower: pl.Series | np.ndarray,
    upper: pl.Series | np.ndarray,
    confidence: float,                # e.g. 0.9
) -> float

# Empirical coverage
coverage_score(
    y_true: pl.Series | np.ndarray,
    lower: pl.Series | np.ndarray,
    upper: pl.Series | np.ndarray,
) -> float                            # fraction of y_true within [lower, upper]
```

---

## 6. Warnings & Errors Reference

| Code | Type | Trigger | Message |
|---|---|---|---|
| `UF-W001` | Warning | `n_calibration < 50` | "Calibration set has only {n} samples. Coverage guarantees may be unreliable." |
| `UF-E001` | Error | `n_calibration < 20` | "Calibration set too small ({n} samples). Minimum is 20." |
| `UF-W002` | Warning | Quantile crossing detected | "Quantile crossing detected in {pct}% of predictions. Post-sort applied. Consider re-evaluating base model quality." |
| `UF-W003` | Warning | Coverage gap > 5% | "Requested {req} coverage but achieved {ach}. Model may be miscalibrated." |
| `UF-W004` | Warning | No uncertainty drivers found | "Residual correlation analysis found no significant drivers. Intervals may be uniformly conservative." |
| `UF-W005` | Warning | LazyFrame materialised early | "LazyFrame collected earlier than expected due to {reason}. Consider restructuring upstream pipeline." |

---

## 7. Common Patterns

### Pattern A: Wrapping an existing model

```python
from sklearn.ensemble import RandomForestRegressor
from uncertainty_flow.wrappers import ConformalRegressor

model = ConformalRegressor(base_model=RandomForestRegressor(n_estimators=200))
model.fit(df_train, target="price")
pred = model.predict(df_test)

print(pred.interval(0.9))
```

### Pattern B: Checking calibration before deploying

```python
report = model.calibration_report(df_val, target="price")
print(report)
# Check if achieved_coverage is within 5% of requested_coverage
assert (report["achieved_coverage"] - report["requested_coverage"]).abs().max() < 0.05
```

### Pattern C: Investigating uncertainty drivers

```python
model.fit(df_train, target="price")
print(model.uncertainty_drivers_)
# feature          residual_correlation   p_value
# volatility       0.71                   0.001
# days_since_event 0.43                   0.012
# region           0.08                   0.34    ← not significant
```

### Pattern D: Multivariate forecasting with joint intervals

```python
from uncertainty_flow.models import ConformalForecaster

model = ConformalForecaster(
    base_model=GradientBoostingRegressor(),
    targets=["price", "volume"],
    horizon=14,
    target_correlation="auto",
)
model.fit(df_train)
pred = model.predict(df_test)

# Joint intervals respect correlation between price and volume
joint = pred.interval(confidence=0.9)
# Columns: price_lower, price_upper, volume_lower, volume_upper
```
