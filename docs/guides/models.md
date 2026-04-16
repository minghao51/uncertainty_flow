# Models Guide

This document is the current guide to model selection and guarantee tradeoffs.

## Guarantee Definitions

| Term | Meaning |
|---|---|
| Mathematical coverage | Interval coverage is backed by a conformal-style guarantee under stated assumptions. |
| Empirical coverage | Coverage must be validated on held-out data and can drift under shift. |
| By construction | Quantile ordering comes from the model form itself. |
| Post-sort | Quantile ordering is repaired after inference rather than enforced by the underlying model. |

## Model Matrix

| Model | Family | Coverage | Non-crossing | Multivariate | Best Fit |
|---|---|---|---|---|---|
| `ConformalRegressor` | Tabular wrapper | Mathematical | Post-sort | No | Add calibrated intervals to an existing regressor |
| `ConformalForecaster` | Forecasting wrapper | Mathematical with temporal assumptions | Post-sort | Yes | Forecasting with calibrated intervals |
| `QuantileForestForecaster` | Native tree model | Empirical | By construction | Yes | Fast, interpretable forecasting |
| `DeepQuantileNet` | Native neural model | Empirical | Post-sort | Yes | Nonlinear patterns with larger datasets |
| `DeepQuantileNetTorch` | Optional neural model | Empirical | Training-time support available | Yes | GPU-backed training and deeper experimentation |
| `TransformerForecaster` | Optional foundation-model wrapper | Empirical or calibrated depending on workflow | Model-dependent | Yes | Pretrained forecasting workflows |
| `BayesianQuantileRegressor` | Bayesian (optional NumPyro) | Posterior-based | Post-sort | No | Full posterior inference, small datasets, credible intervals |
| `CausalUncertaintyEstimator` | Causal inference | Conformal on CATE | N/A | No | Treatment effect estimation with uncertainty |
| `CrossModalAggregator` | Multi-modal ensemble | Inherited from base models | Inherited | Yes | Combine predictions from separate feature groups |
| `ConformalRiskControl` | Risk control | Risk-bounded | Post-sort | No | Control expected loss instead of coverage |

## Shared Output Contract

Every model returns `DistributionPrediction`, so downstream usage stays consistent:

```python
pred = model.predict(df_test)
pred.quantile([0.1, 0.5, 0.9])
pred.interval(0.9)
pred.mean()
pred.sample(100)
model.save("models/example.uf")
```

That shared surface was previously described in a separate guide and is intentionally consolidated here.

For uncertainty decomposition, there are now two levels of fidelity:

- `pred.uncertainty_decomposition(...)` is a lightweight heuristic computed from one prediction object.
- `EnsembleDecomposition(model_factory=..., train_data=...)` performs bootstrap refits for a stronger epistemic estimate.
- Multivariate feature leverage currently reports per-target attribution only; joint copula leverage is not part of the current surface.

## Model Notes

### `ConformalRegressor`

Use this when you already have a scikit-learn compatible regressor and want calibrated intervals on tabular data. The guarantee depends on exchangeability and a sufficiently large calibration set.

### `ConformalForecaster`

Use this when forecasting data is ordered and interval calibration matters more than raw speed. It supports multi-target forecasting through the multivariate layer, and multivariate `sample()` calls respect the fitted copula after prediction.

### `QuantileForestForecaster`

Use this when you want a solid non-deep-learning baseline with fast quantile retrieval and sensible behavior on moderate-sized datasets. Its `copula_family` setting is active for multivariate targets and feeds copula-aware joint sampling.

### `DeepQuantileNet` and `DeepQuantileNetTorch`

Use these when the signal is nonlinear enough that tree models underfit. The torch-backed path is the better fit when you want a more flexible training loop or accelerator support.

### `TransformerForecaster`

Use this for pretrained time-series workflows when the optional dependency stack is available and the project favors foundation-model-style forecasting.

### `BayesianQuantileRegressor`

Use this when you need full posterior distributions rather than discrete quantiles. Provides credible intervals on quantiles themselves via MCMC (NUTS sampler). Best suited for smaller datasets where conformal methods may lack calibration data. Requires `numpyro` and `jax` (optional dependency).

### `CausalUncertaintyEstimator`

Use this to estimate treatment effects (CATE/ATE) with conformal confidence intervals. Supports doubly-robust, S-learner, and T-learner methods. Wraps outcome and propensity models built from existing `ConformalRegressor` instances. No extra dependencies.

### `CrossModalAggregator`

Use this when features naturally form groups (demographics, temporal, weather) and you want per-group uncertainty attribution alongside combined predictions. Aggregation strategies: product (assumes conditional independence), copula (models cross-group dependence), independent (simple average). No extra dependencies.

### `ConformalRiskControl`

Use this when prediction errors have asymmetric costs. Instead of controlling coverage, calibrates intervals to control expected risk for user-defined loss functions. Ships with built-in risk functions: `asymmetric_loss`, `threshold_penalty`, `inventory_cost`, `financial_var`. No extra dependencies.

## Multivariate Dependence

Multi-target models rely on a copula layer rather than treating targets as independent. Depending on the workflow, this may use Gaussian, Clayton, Gumbel, or Frank families, with auto-selection in supported paths.

## Persistence

All current model classes inherit a shared persistence contract:

```python
model.save("models/example.uf")
loaded = type(model).load("models/example.uf")
```

Saved archives preserve fitted model state, calibration artifacts, and multivariate copula state. In trusted environments this is the recommended way to move fitted models between sessions.

## Verification

For a focused persistence and copula round-trip check, run:

```bash
uv run pytest tests/core/test_persistence.py
```

## Choosing Quickly

- Need a mathematical interval guarantee for tabular regression: `ConformalRegressor`
- Need a mathematical interval guarantee for forecasting: `ConformalForecaster`
- Need fast empirical quantiles without deep learning: `QuantileForestForecaster`
- Need higher-capacity nonlinear modeling: `DeepQuantileNet` or `DeepQuantileNetTorch`
- Need optional pretrained forecasting workflows: `TransformerForecaster`
- Need full posterior inference or credible intervals: `BayesianQuantileRegressor`
- Need treatment effect estimates with uncertainty: `CausalUncertaintyEstimator`
- Need to combine predictions from feature groups: `CrossModalAggregator`
- Need to control expected loss instead of coverage: `ConformalRiskControl`
- Need to find feature changes that reduce uncertainty: `UncertaintyExplainer` (see [./charting.md](./charting.md))
- Need to decompose uncertainty into aleatoric vs epistemic: `EnsembleDecomposition` (see [./distribution-approach.md](./distribution-approach.md))
- Need to score features by impact on interval width: `FeatureLeverageAnalyzer` (see [./distribution-approach.md](./distribution-approach.md))
