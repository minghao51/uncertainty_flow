# API_SPEC.md — Full API Specification

---

## Overview

All public classes follow a consistent `fit` / `predict` interface. Inputs are Polars DataFrames or LazyFrames. Outputs are `DistributionPrediction` objects or Polars DataFrames.

All uncertainty models also inherit:

```python
model.save("models/example.uf", include_metadata=True)
loaded = ModelClass.load("models/example.uf")
model.metadata  # dict for fitted or loaded models, else None
```

`load()` deserializes model payloads with Python pickle. Only load `.uf` archives from trusted sources.

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
        Returns DistributionPrediction with the quantile levels captured at fit time.
        If global quantile config changes after fit, prediction still uses fitted levels.
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

    def predict_batch(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        batch_size: int = 1000,
    ) -> Iterator[DistributionPrediction]:
        """
        Memory-batched prediction. Yields DistributionPrediction
        chunks of up to batch_size rows each.
        """
        ...

    def save(
        self,
        path: str,
        include_metadata: bool = True,
    ) -> str:
        """
        Serializes model to a .uf archive.
        Returns the path written to.
        """
        ...

    @staticmethod
    def load(path: str) -> "ConformalRegressor":
        """
        Deserializes a .uf archive.
        Only load from trusted sources (uses pickle).
        """
        ...

    def analyze_leverage(
        self,
        data: pl.DataFrame,
        confidence: float = 0.9,
    ) -> pl.DataFrame:
        """
        Per-sample feature leverage analysis using SHAP.
        Returns DataFrame with feature importance scores for
        interval width at each sample.
        """
        ...

    def explain_interval_width(
        self,
        data: pl.DataFrame,
        confidence: float = 0.9,
    ) -> pl.DataFrame:
        """
        Explain interval width predictions for each sample.
        Aggregates SHAP values into per-feature width contribution.
        """
        ...```

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
        copula_family: str = "auto",         # "auto" | "gaussian" | "clayton"
                                             # | "gumbel" | "frank" | "independent"
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
        Fits a supported copula on residuals if copula_family='auto' and
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

    def predict_batch(self, data, batch_size=1000) -> Iterator[DistributionPrediction]: ...

    def save(self, path, include_metadata=True) -> str: ...

    @staticmethod
    def load(path: str) -> "ConformalForecaster": ...

    def analyze_leverage(self, data, confidence=0.9) -> pl.DataFrame: ...

    def explain_interval_width(self, data, confidence=0.9) -> pl.DataFrame: ...
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
        copula_family: str = "auto",
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

    def predict_batch(self, data, batch_size=1000) -> Iterator[DistributionPrediction]: ...

    def save(self, path, include_metadata=True) -> str: ...

    @staticmethod
    def load(path: str) -> "QuantileForestForecaster": ...

    def analyze_leverage(self, data, confidence=0.9) -> pl.DataFrame: ...

    def explain_interval_width(self, data, confidence=0.9) -> pl.DataFrame: ...
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
        Draw n samples per input row via inverse-CDF sampling.
        For multivariate predictions with attached copula state, sampling respects
        the fitted copula rather than treating targets as independent.
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

    def summary(
        self,
        confidence: float = 0.9,
    ) -> pl.DataFrame:
        """
        One-row summary DataFrame with columns: target, median,
        mean_width_90, mean_width_50, aleatoric, epistemic, total_uncertainty.
        """

    def energy_score(
        self,
        y_true,
        n_samples: int = 1000,
        random_state: int | None = None,
    ) -> float:
        """
        Multivariate energy score. Draws two independent samples from
        the predicted distribution and computes E[‖X-y‖] - ½E[‖X-X'‖].
        For univariate, equivalent to CRPS.
        """

    def variogram_score(
        self,
        y_true,
        n_samples: int = 1000,
        p: float = 1.0,
        random_state: int | None = None,
    ) -> float:
        """
        Variogram score of order p. Assesses multivariate joint
        dependence by comparing weighted distances between targets.
        """

    def log_score(
        self,
        y_true,
        family: str = "normal",
    ) -> float:
        """
        Log-score (negative log-likelihood) of true values under
        a parametric fit. Supported families: normal, student_t,
        logistic, gumbel, laplace, cauchy.
        """

    def crps(
        self,
        y_true,
    ) -> float:
        """
        Continuous Ranked Probability Score. Integrated quantile loss
        across all fitted quantile levels.
        """

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
from uncertainty_flow.metrics import (
    pinball_loss, winkler_score, coverage_score,
    crps_quantile, crps_score,
    log_score, log_score_kde, log_score_pooled,
    energy_score, variogram_score,
    skill_score, diebold_mariano_test, model_confidence_set,
    mae_score, rmse_score,
    calibration_error,
)

# Unified entry point — dispatches by metric name
from uncertainty_flow.metrics import score
score(pred, y_true, metric="crps")          # CRPS
score(pred, y_true, metric="log_score")     # Log-score
score(pred, y_true, metric="energy_score")  # Energy score
score(pred, y_true, metric="coverage")      # Empirical coverage
score(pred, y_true, metric="winkler")       # Winkler interval score
score(pred, y_true, metric="pinball")       # Mean pinball loss across levels

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

# CRPS (quantile approximation)
crps_score(
    y_true: pl.Series | np.ndarray,
    pred: DistributionPrediction,     # Or (quantile_matrix, quantile_levels)
) -> float

# Log-score (log-likelihood)
log_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family: str = "normal",           # normal, student_t, logistic, gumbel, laplace, cauchy
) -> float

# Energy score (multivariate)
energy_score(
    y_true: np.ndarray,
    y_pred_samples: np.ndarray,       # (n_samples, n_targets) or (n, n_samples, n_targets)
) -> float

# Variogram score (multivariate dependence)
variogram_score(
    pred: DistributionPrediction,
    y_true: np.ndarray | pl.DataFrame,
    p: float = 1.0,
    n_samples: int = 1000,
) -> float

# Model comparison
skill_score(
    pred_a: DistributionPrediction,
    pred_b: DistributionPrediction,
    y_true,                           # Ground truth
    metric: str = "crps",             # Metric name
) -> float                            # Positive = pred_a better than pred_b

diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> dict[str, float]                 # DM statistic and p-value

model_confidence_set(
    predictions: list[DistributionPrediction],
    y_true,
) -> pl.DataFrame                     # Which models survive the MCS procedure
```

---

## 6. Warnings & Errors Reference

| Code | Type | Trigger | Message |
|---|---|---|---|---|
| `UF-W001` | Warning | `n_calibration < 50` | "Calibration set has only {n} samples. Coverage guarantees may be unreliable." |
| `UF-E001` | Error | `n_calibration < 20` | "Calibration set too small ({n} samples). Minimum is 20." |
| `UF-E002` | Error | Model not fitted | "{ModelName} not fitted. Call .fit() first." |
| `UF-E003` | Error | Invalid input data | "Invalid data: {reason}" |
| `UF-E004` | Error | Quantile config invalid | "Invalid quantile configuration: {reason}" |
| `UF-W002` | Warning | Quantile crossing detected | "Quantile crossing detected in {pct}% of predictions. Post-sort applied. Consider re-evaluating base model quality." |
| `UF-W003` | Warning | Coverage gap > 5% | "Requested {req} coverage but achieved {ach}. Model may be miscalibrated." |
| `UF-W004` | Warning | No uncertainty drivers found | "Residual correlation analysis found no significant drivers. Intervals may be uniformly conservative." |
| `UF-W005` | Warning | LazyFrame materialised early | "LazyFrame collected earlier than expected due to {reason}. Consider restructuring upstream pipeline." |
| `UF-W006` | Warning | Copula auto-select with dim > 2 | "Auto-selecting copula for {n_dim}D data. Only Gaussian copula supports dimensions > 2." |

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
    copula_family="auto",
)
model.fit(df_train)
pred = model.predict(df_test)

# Joint intervals respect correlation between price and volume
joint = pred.interval(confidence=0.9)
# Columns: price_lower, price_upper, volume_lower, volume_upper
```

---

## 8. `DeepQuantileNet`

> Multi-quantile MLP with shared trunk (sklearn backend).
> **Coverage guarantee: ⚠️ Empirical only**
> **Non-crossing: ✅ (post-sort)**

```python
class DeepQuantileNet(BaseQuantileNeuralNet, RegressorMixin):

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100, 50),
        quantile_levels: list[float] | None = None,
        trunk_alpha: float = 0.0001,
        trunk_max_iter: int = 500,
        head_solver: str = "pinball",
        non_crossing_penalty: float = 0.1,   # Weight added to loss for quantile crossing penalty
        random_state: int | None = None,
    ): ...
```

---

## 9. `DeepQuantileNetTorch`

> PyTorch-backed multi-quantile network with GPU support and optional monotonicity loss.
> **Coverage guarantee: ⚠️ Empirical only**
> **Non-crossing: ✅ (training-time support via monotonicity_weight)**
> **Requires:** `torch` (optional dependency)

```python
class DeepQuantileNetTorch(BaseQuantileNeuralNet):

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100, 50),
        quantile_levels: list[float] | None = None,
        n_estimators: int = 1,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        monotonicity_weight: float = 0.1,
        activation: str = "relu",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ): ...
```

---

## 10. `TransformerForecaster`

> Pretrained foundation-model forecasting wrapper (Chronos-2 integration).
> **Coverage guarantee: ⚠️ Empirical or calibrated depending on workflow**
> **Requires:** `chronos-forecasting` (optional dependency)

```python
class TransformerForecaster(BaseUncertaintyModel):

    def __init__(
        self,
        target: str,
        horizon: int = 24,
        model_name: str | None = None,
        calibration_method: str = "holdout",
        calibration_size: float = 0.2,
        auto_tune: bool = True,
        device: str = "auto",
        random_state: int | None = None,
        uncertainty_features: list[str] | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target: str | None = None,
    ) -> "TransformerForecaster": ...

    def predict(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        steps: int | None = None,
    ) -> DistributionPrediction: ...

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None: ...
```

---

## 11. `BayesianQuantileRegressor`

> Bayesian quantile regression via NumPyro MCMC with horseshoe priors.
> **Coverage guarantee: Posterior-based (credible intervals, not frequentist coverage)**
> **Requires:** `numpyro`, `jax` (optional dependency)

```python
class BayesianQuantileRegressor(BaseUncertaintyModel):

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_warmup: int = 500,
        n_samples: int = 1000,
        num_chains: int = 1,
        kernel: str = "nuts",
        prior_width: float = 1.0,
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target: str | None = None,
    ) -> "BayesianQuantileRegressor": ...

    def predict(self, data: pl.DataFrame | pl.LazyFrame) -> DistributionPrediction: ...
```

---

## 12. `CausalUncertaintyEstimator`

> Treatment effect estimation. Supports doubly-robust, S-learner, and T-learner methods.
> `predict()` is label-free; outcome-dependent ATE/CI metrics are computed with `evaluate(...)`.

```python
class CausalUncertaintyEstimator(BaseUncertaintyModel):

    def __init__(
        self,
        outcome_model,                      # sklearn-like regressor; DR mode disallows conformal wrappers
        propensity_model=None,              # Optional, defaults to logistic
        treatment_col: str = "treatment",
        method: str = "doubly_robust",      # "doubly_robust" | "s_learner" | "t_learner"
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target: str | None = None,
    ) -> "CausalUncertaintyEstimator": ...

    def predict(self, data: pl.DataFrame | pl.LazyFrame) -> DistributionPrediction: ...

    def evaluate(self, data: pl.DataFrame | pl.LazyFrame) -> dict: ...
```

---

## 13. `CrossModalAggregator`

> Combine predictions from multiple feature groups with per-group uncertainty attribution.

```python
class CrossModalAggregator(BaseUncertaintyModel):

    def __init__(
        self,
        feature_groups: dict[str, list[str]],  # {"demographics": ["age", ...], ...}
        aggregation: str = "product",           # "product" | "copula" | "independent"
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target: str | None = None,
        *,
        base_model=None,                       # Base model for each group
    ) -> "CrossModalAggregator": ...

    def predict(self, data: pl.DataFrame | pl.LazyFrame) -> DistributionPrediction: ...
```

`aggregation="copula"` is currently reserved and raises `NotImplementedError`.

---

## 14. `ConformalRiskControl`

> Conformal risk control — calibrates intervals to control expected risk rather than coverage.

```python
class ConformalRiskControl:

    def __init__(
        self,
        base_model: BaseUncertaintyModel,
        risk_function: Callable,               # (y_true, y_pred) → risk scalar
        target_risk: float = 0.1,
        calibration_method: str = "quantile",
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame,
        target: str,
    ) -> "ConformalRiskControl": ...

    def predict(self, data: pl.DataFrame) -> pl.DataFrame: ...

    def risk_threshold(self) -> float: ...

    def summary(self) -> dict: ...
```

### Built-in Risk Functions

```python
from uncertainty_flow.risk import asymmetric_loss, threshold_penalty, inventory_cost, financial_var

# Asymmetric over/underprediction penalty
asymmetric_loss(overprediction_penalty=1.0, underprediction_penalty=2.0)

# Penalty above/below a threshold
threshold_penalty(threshold=0.0, penalty_above=10.0, penalty_below=1.0)

# Inventory holding vs stockout cost
inventory_cost(holding_cost=1.0, stockout_cost=10.0)

# Value-at-Risk style penalty
financial_var(var_level=0.95)
```

---

## 15. `UncertaintyExplainer`

> Counterfactual explanations for uncertainty reduction. Finds minimal feature changes that reduce interval width.

```python
class UncertaintyExplainer:

    def __init__(
        self,
        model: BaseUncertaintyModel,
        confidence: float = 0.9,
        method: str = "auto",                  # "auto" | "evolutionary" | "gradient"
        random_state: int | None = None,
    ): ...

    def explain_uncertainty(
        self,
        data: pl.DataFrame,
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        fixed_features: list[str] | None = None,
    ) -> SearchResult: ...

    def explain_batch(
        self,
        data: pl.DataFrame,
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
        fixed_features: list[str] | None = None,
    ) -> list[SearchResult]: ...

    def compare_features(
        self,
        data: pl.DataFrame,
        features: list[str],
        target_reduction: float = 0.5,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> pl.DataFrame: ...
```

---

## 16. `EnsembleDecomposition`

> Bootstrap-based aleatoric/epistemic uncertainty decomposition.

```python
class EnsembleDecomposition:

    def __init__(
        self,
        model_factory: Callable[[], BaseUncertaintyModel],
        train_data: pl.DataFrame | pl.LazyFrame,
        target: str | None = None,
        confidence: float = 0.9,
        n_bootstrap: int = 5,
        random_state: int | None = None,
    ): ...

    def decompose(self, data: pl.DataFrame) -> dict[str, float]:
        """
        Returns:
            aleatoric: average interval width across ensemble
            epistemic: variance of point estimates across ensemble
            total: combined uncertainty
        """

    def decompose_by_sample(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Returns per-sample decomposition as Polars DataFrame.
        Columns: aleatoric, epistemic, total
        """

    def summary(self) -> dict: ...
```

---

## 17. `FeatureLeverageAnalyzer`

> Scores features by their impact on prediction interval width, separating aleatoric from epistemic contributions.

```python
class FeatureLeverageAnalyzer:

    def __init__(
        self,
        model: BaseUncertaintyModel,
        confidence: float = 0.9,
        n_perturbations: int = 100,
        n_bins: int = 10,
        leverage_threshold: float = 0.5,
        random_state: int | None = None,
    ): ...

    def analyze(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Returns Polars DataFrame schema:
        ┌──────────────┬─────────────────┬─────────────────┬──────────────┬────────────────┐
        │ feature      │ aleatoric_score │ epistemic_score │ leverage_score│ recommendation │
        │ str          │ f64             │ f64             │ f64          │ str            │
        └──────────────┴─────────────────┴─────────────────┴──────────────┴────────────────┘
        """

    def analyze_multivariate(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Per-target leverage analysis for multivariate models.
        Columns: feature, {target}_aleatoric, {target}_epistemic, {target}_leverage, recommendation
        """

    def summary(self) -> dict: ...

---

## 18. `AdaptiveConformalForecaster`

> Time-adaptive conformal inference for streaming/online settings. Adjusts miscoverage rate $\alpha_t$ after each true-value observation.
> **Coverage guarantee: ✅ (asymptotic, under bounded temporal drift)**
> **Non-crossing: ✅ (inherits from base model)**

```python
class AdaptiveConformalForecaster(BaseUncertaintyModel):

    def __init__(
        self,
        model: BaseUncertaintyModel,       # Pre-fitted base model
        alpha: float = 0.1,                # Target miscoverage rate (0, 1)
        gamma: float = 0.01,               # Learning rate for alpha adaptation
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame,
        target: str,
    ) -> "AdaptiveConformalForecaster":
        """
        Initializes conformal scores and alpha_0 from calibration split.
        Requires target for score computation.
        """
        ...

    def predict(
        self,
        data: pl.DataFrame,
    ) -> DistributionPrediction:
        """
        Returns intervals using current alpha_t.
        Stores per-target point predictions for subsequent update().
        """
        ...

    def update(
        self,
        y_true: float | np.ndarray,
    ) -> float:
        """
        Adjusts alpha_t after observing true value.
        Accepts scalar (univariate) or array (multivariate).
        Validates dimension matches number of targets.
        Returns the new alpha value.
        """
        ...

    def update_batch(
        self,
        y_true: pl.Series | np.ndarray,
    ) -> float:
        """
        Batch-update with multiple true values (one per predict step).
        """
        ...
```

---

## 19. `EnsembleBootstrapPI` (EnbPI)

> Ensemble Bootstrap Prediction Intervals (Xu & Xie 2021). Combines B bootstrap base learners with sequential conformal score updates. Designed for time series.

```python
class EnsembleBootstrapPI(BaseUncertaintyModel):

    def __init__(
        self,
        base_model_factory: Callable[[], Any],  # Returns a fresh sklearn estimator
        n_models: int = 20,                     # B bootstrap models
        coverage_target: float = 0.9,
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target: str,
    ) -> "EnsembleBootstrapPI":
        """
        Trains B bootstrap models on bootstrapped samples.
        Stores out-of-bag residuals for conformal initialization.
        """
        ...

    def predict(
        self,
        data: pl.DataFrame | pl.LazyFrame,
    ) -> DistributionPrediction:
        """
        Aggregates bootstrap predictions. Returns intervals via
        sequential conformal scores if update() has been called.
        """
        ...
```

---

## 20. `ConformalClassifier` & `PredictionSet`

> Conformal classification via Adaptive Prediction Sets (APS). Unlike regression models, classification outputs `PredictionSet` objects.

```python
class ConformalClassifier:

    def __init__(
        self,
        base_model,                          # sklearn classifier with predict_proba
        coverage_target: float = 0.9,
        random_state: int | None = None,
    ): ...

    def fit(
        self,
        data: pl.DataFrame,
        target: str,
    ) -> "ConformalClassifier":
        """
        Fits base classifier on training data.
        Computes conformity scores (softmax probabilities) on calibration split.
        """
        ...

    def predict(
        self,
        data: pl.DataFrame,
    ) -> "PredictionSet":
        """
        Returns PredictionSet for each row: the smallest set of labels
        whose cumulative softmax probability exceeds the threshold.
        """
        ...

    def predict_batch(
        self,
        data: pl.DataFrame,
        batch_size: int = 1000,
    ) -> Iterator["PredictionSet"]: ...

    def save(self, path: str, include_metadata: bool = True) -> str: ...

    @staticmethod
    def load(path: str) -> "ConformalClassifier": ...


class PredictionSet:

    def __init__(self, ...): ...

    def set(self) -> list[set]:
        """Return the prediction set for each row as a Python set of labels."""

    def coverage(self) -> float:
        """Average set size across all rows."""

    def size(self) -> list[int]:
        """Number of labels in each prediction set."""
```

---

## 21. `ParametricDistribution`

> Fit parametric distributions to data and compute log-likelihood, quantiles, CRPS, and sampling.

```python
class ParametricDistribution:

    def __init__(
        self,
        family: str,                          # normal, student_t, logistic, gumbel,
                                              # laplace, cauchy, beta, gamma, lognormal
        params: dict[str, float] | None = None,
    ): ...

    def fit(self, data: np.ndarray) -> "ParametricDistribution": ...

    def quantile(self, q: float | np.ndarray) -> np.ndarray: ...

    def log_likelihood(self, data: np.ndarray) -> float: ...

    def crps(self, data: np.ndarray) -> float: ...

    def sample(self, n: int, random_state: int | None = None) -> np.ndarray: ...


# Standalone helper
def fit_parametric(
    data: np.ndarray,
    family: str = "normal",
) -> ParametricDistribution: ...
```

---

## 22. Copula Families

> Multivariate dependence modeling. Extended with `PairwiseChainCopula` for >2 targets.

```python
COPULA_FAMILIES = {
    "gaussian": GaussianCopula,
    "clayton": ClaytonCopula,
    "gumbel": GumbelCopula,
    "frank": FrankCopula,
    "pairwise_chain": PairwiseChainCopula,
}


class PairwiseChainCopula(BaseCopula):

    def __init__(self, base_family: str = "gaussian"): ...

    def fit(self, residuals: np.ndarray) -> "PairwiseChainCopula":
        """
        Decomposes d-dimensional residuals into d-1 bivariate copulas
        via a D-vine-like chain structure. Requires at least 2 targets.
        """
        ...

    def sample(
        self,
        marginals: np.ndarray,              # (n, d, n_levels) uniform quantiles
        n_samples: int,
        quantile_levels: np.ndarray,
        random_state: int | None = None,
    ) -> np.ndarray:                        # (n, n_samples, d)
        ...

    def log_likelihood(self, residuals: np.ndarray) -> float: ...


def auto_select_copula(residuals: np.ndarray) -> str:
    """
    Auto-selects copula family by BIC. Considers pairwise_chain
    for dimensions >= 3 alongside gaussian.
    """
    ...
```
